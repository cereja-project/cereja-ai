import calango
import numpy as np
import cv2

__all__ = ['VideoKeypointsExtractor']
MAPED_FACE_LANDMARKS = [398, 384, 385, 386, 387, 388, 466, 362, 382, 381, 380, 374, 373, 390, 249]  # left eye
MAPED_FACE_LANDMARKS += [246, 7, 161, 160, 159, 158, 157, 173, 163, 144, 145, 153, 154, 155]  # right eye


def _mediapipe_generator(draw=False):
    import mediapipe as mp
    import cv2
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
        while True:
            image = yield
            if image is None:
                break
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            # Draw landmark annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                pose = [(i.x, i.y, i.z) for i in results.pose_landmarks.landmark]
                # if draw:
                #     mp_drawing.draw_landmarks(
                #             image,
                #             results.pose_landmarks,
                #             mp_holistic.POSE_CONNECTIONS)
            else:
                pose = [[0.0, 0.0, 0.0]] * 33

            if results.left_hand_landmarks:
                left_hand = [(i.x, i.y, i.z) for i in results.left_hand_landmarks.landmark]
                if draw:
                    mp_drawing.draw_landmarks(
                            image,
                            results.left_hand_landmarks,
                            mp_holistic.HAND_CONNECTIONS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_hand_connections_style())
            else:
                left_hand = [[0.0, 0.0, 0.0]] * 21

            if results.right_hand_landmarks:
                right_hand = [(i.x, i.y, i.z) for i in results.right_hand_landmarks.landmark]
                if draw:
                    mp_drawing.draw_landmarks(
                            image,
                            results.right_hand_landmarks,
                            mp_holistic.HAND_CONNECTIONS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_hand_connections_style())
            else:
                right_hand = [[0.0, 0.0, 0.0]] * 21

            if results.face_landmarks:
                face_data = [[i.x, i.y, i.z] for indx, i in enumerate(results.face_landmarks.landmark) if
                             indx in MAPED_FACE_LANDMARKS]
                if draw:
                    mp_drawing.draw_landmarks(
                            image,
                            results.face_landmarks,
                            mp_holistic.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style())
            else:
                face_data = [[0.0, 0.0, 0.0]] * 14
            yield image, [face_data, pose, left_hand, right_hand]


class VideoKeypointsExtractor:
    def __init__(self, *args, draw=False):
        self._stopped = False
        self._extractor = _mediapipe_generator(draw=draw)
        self._video = calango.Video(*args, fps=None)

    @property
    def is_extracting(self):
        return self._video.is_opened

    def _extract_all(self, show=False, text_bottom=None) -> np.ndarray:
        self._stopped = False
        # keypoints = []
        try:
            while self._video.is_opened:
                next(self._extractor)
                image = self._video.next_frame
                if image is None:
                    self._extractor.send(None)
                    break
                image, kpts = self._extractor.send(image.flip())

                if show:
                    cv2.imshow("Video", image if text_bottom is None else calango.Image(image).write_text(text_bottom))
                    if self._video.is_break_view:
                        self._video.stop()
                # keypoints.append(kpts)
                yield kpts
        except StopIteration:
            self._stop()
        except Exception as err:
            self._stop()
            raise Exception(f"error during capture: {err}")
        self._stop()
        # return np.array(keypoints)

    def extract_all(self, show=False):
        face_data = []
        pose_data = []
        hand_l_data = []
        hand_r_data = []
        for face, pose, hl, hr in self._extract_all(show=show):
            face_data.append(face)
            pose_data.append(pose)
            hand_l_data.append(hl)
            hand_r_data.append(hr)
        return face_data, pose_data, hand_l_data, hand_r_data

    def _stop(self):
        if not self._stopped:
            try:
                next(self._extractor)
                self._extractor.send(None)
            except:
                pass
            cv2.destroyAllWindows()
            self._stopped = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop()
