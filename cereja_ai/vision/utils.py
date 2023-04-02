import calango
import numpy as np
import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

__all__ = ['VideoKeypointsExtractor', 'SequenceArray']
MAPED_FACE_LANDMARKS = [398, 384, 385, 386, 387, 388, 466, 362, 382, 381, 380, 374, 373, 390, 249]  # left eye
MAPED_FACE_LANDMARKS += [246, 7, 161, 160, 159, 158, 157, 173, 163, 144, 145, 153, 154, 155]  # right eye


class SequenceArray:
    def __init__(self, arr, dtype=None):
        arr = np.array(arr, dtype=dtype)
        self._default_shape = arr.shape[1:]
        self._data = arr

    @classmethod
    def empty(cls, shape, dtype=None):
        return cls(np.empty([0, *shape], dtype=dtype))

    @property
    def is_empty(self):
        return len(self.data) == 0

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._data.shape

    def clean(self):
        """
        Turn empty.
        """
        self._data = np.empty([0, *self._default_shape], dtype=self._data.dtype)

    def append(self, new_data):
        """
        Similar to the append method of a list
        """
        self._data = np.vstack([self._data, self._check_and_parse(new_data)])

    def pop_batch(self, batch_size):
        batch = self[:batch_size]
        self._data = self[batch_size:]
        return batch

    @staticmethod
    def _split_sequence_on_breaks(seq):
        indices = np.where(np.diff(seq) != 1)[0] + 1
        sub_seqs = np.split(seq, indices)
        return sub_seqs

    def interpolate(self):
        """
        Fills empty values in a 3D coordinate string.
        """

        # Identifies points with missing values
        has_missing_values = np.any(self._data == 0, axis=-1)
        if not np.any(has_missing_values):
            return
        idxs = np.unique(np.where(has_missing_values)[0])
        # Break sequence when next is not subsequent
        idxs = self._split_sequence_on_breaks(idxs)

        # Interpolate the missing values
        for seq in idxs:
            # If the first frame is zero, we take the next valid one
            # TODO: Check if it makes sense to get the last frame that can be rest. Or set the resting kpts to not start at zero.
            last_valid = seq[0] - 1 if seq[0] > 0 else seq[-1] + 1
            next_valid = seq[-1] + 1
            self._data[seq,] = np.linspace(self._data[last_valid if len(self._data) > last_valid else last_valid - 1],
                                           self._data[next_valid if len(self._data) > next_valid else next_valid - 1],
                                           len(seq))

    def _check_and_parse(self, v):
        if isinstance(v, (list, tuple, np.ndarray)):
            v = np.array(v, ndmin=len(self.shape))
        else:
            raise TypeError("Invalid data type.")
        _shape = v.shape
        assert self.shape[1:] == _shape[1:], f"Format {_shape} it is invalid. Expected {self.shape}"
        return v

    def __str__(self):
        return self._data.__str__()

    def __repr__(self):
        return self._data.__repr__()

    def __getitem__(self, item):
        return SequenceArray(self._data.__getitem__(item))

    def __len__(self):
        return self.data.__len__()

    def __iter__(self):
        return self._data.__iter__()

    def __eq__(self, other):
        return self._data.__eq__(other)

    def __matmul__(self, other):
        return SequenceArray(self._data.__matmul__(other))

    def __add__(self, other):
        return SequenceArray(self._data.__add__(other))

    def __sub__(self, other):
        return SequenceArray(self._data.__sub__(other))

    def __mul__(self, other):
        return SequenceArray(self._data.__mul__(other))

    def __rmul__(self, other):
        return SequenceArray(self._data.__rmul__(other))

    def __truediv__(self, other):
        return SequenceArray(self._data.__truediv__(other))

    def __iadd__(self, other):
        return SequenceArray(self._data.__iadd__(other))


class VideoKeypointsSequence:
    R_WRIST = 16
    L_WRIST = 15

    def __init__(self, face, pose, hand_l, hand_r):
        if not isinstance(face, SequenceArray):
            face = SequenceArray(face)
            pose = SequenceArray(pose)
            hand_l = SequenceArray(hand_l)
            hand_r = SequenceArray(hand_r)
        self._face = face
        self._pose = pose
        self._hand_l = hand_l
        self._hand_r = hand_r

    @classmethod
    def empty(cls, face_shape, pose_shape, hand_l_shape, hand_r_shape):
        return cls(SequenceArray.empty(face_shape),
                   SequenceArray.empty(pose_shape),
                   SequenceArray.empty(hand_l_shape),
                   SequenceArray.empty(hand_r_shape))

    @property
    def hand_l(self):
        return self._hand_l

    @property
    def hand_r(self):
        return self._hand_r

    @property
    def pose(self):
        return self._pose

    @property
    def face(self):
        return self._face

    @property
    def data(self):
        self.hand_r.interpolate()
        self.hand_l.interpolate()
        self.pose.interpolate()
        # self.face.interpolate()
        self.put_hand_in_pose()
        return np.concatenate([self.pose, self.hand_l, self.hand_r], axis=1)

    def put_hand_in_pose(self):
        wrist_left_pose = self.pose[:, self.L_WRIST:self.L_WRIST + 1]  # output [seq_len, 1, 3]
        wrist_right_pose = self.pose[:, self.R_WRIST:self.R_WRIST + 1]  # output [seq_len, 1, 3]

        # 0 on hand data is wrist
        wrist_left_hand, wrist_right_hand = self.hand_l[:, 0:1], self.hand_r[:, 0:1]

        # move all relative to wrist_pose.
        self.hand_l.data[:, ] = (self.hand_l - wrist_left_hand) + wrist_left_pose
        self.hand_r.data[:, ] = (self.hand_r - wrist_right_hand) + wrist_right_pose


def _mediapipe_generator(draw=False):
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
                face_data = [[0.0, 0.0, 0.0]] * len(MAPED_FACE_LANDMARKS)
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
        return None, None, None

    def extract_all(self, show=False):
        video_kpts = VideoKeypointsSequence.empty([len(MAPED_FACE_LANDMARKS), 3], [33, 3], [21, 3], [21, 3])
        for face, pose, hl, hr in self._extract_all(show=show):
            if all([face is None, pose is None, hl is None, hr is None]):
                break
            video_kpts.face.append(face)
            video_kpts.pose.append(pose)
            video_kpts.hand_l.append(hl)
            video_kpts.hand_r.append(hr)
        return video_kpts.data

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
