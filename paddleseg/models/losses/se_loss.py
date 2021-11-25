# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle import nn
from paddle.framework import dtype
import paddle.nn.functional as F
from paddleseg.cvlibs import manager

@manager.LOSSES.add_component
class SELoss(nn.Layer):
    """
    Implements the se loss function.

    Args:
        ignore_index (int64): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self, ignore_index=255, num_classes=19):
        super(SELoss, self).__init__()
        self.ignore_index = ignore_index
        self.num_classes = num_classes
    @staticmethod
    def _convert_to_onehot_labels(seg_label, num_classes):
        """Convert segmentation label to onehot.

        Args:
            seg_label (Tensor): Segmentation label of shape (N, H, W).
            num_classes (int): Number of classes.

        Returns:
            Tensor: Onehot labels of shape (N, num_classes).
        """

        batch_size = seg_label.shape[0]
        onehot_labels = paddle.zeros((batch_size, num_classes), dtype='bool')
        for i in range(batch_size):
            hist = paddle.histogram(seg_label[i], bins=num_classes, min=0, max=num_classes - 1)
            onehot_labels[i] = hist > 0
        onehot_labels = paddle.cast(onehot_labels, 'float32')
        return onehot_labels

    def forward(self, logits, labels):
        labels = paddle.cast(labels, dtype='int32')
        mask = (paddle.unsqueeze(labels, 1) != self.ignore_index)
        labels = labels * mask
        labels = paddle.cast(labels, dtype='float32')
        
        labels = self._convert_to_onehot_labels(labels, self.num_classes)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction='mean',
            weight=None)
        return loss