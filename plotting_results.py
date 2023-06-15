from util.plot_utils import plot_logs

from pathlib import Path

import matplotlib.pyplot as plt
log_directory = [Path('logdirs/modalFT/')]

# As mentioned in the code of plot_logs:
    # solid lines are training results,
    # dashed lines are validation results.

# fields_of_interest = (
#     'loss',
#     'mAP',
#     )

# plot_logs(log_directory,
#           fields_of_interest)

fields_of_interest = (
    'class_error',
    'loss_bbox',
    'loss_giou',
    'loss'
    )
#
plot_logs(log_directory,
          fields_of_interest)
# #
# fields_of_interest = (
#     'class_error',
#     'cardinality_error_unscaled',
#     )
#
# plot_logs(log_directory,
#           fields_of_interest)
plt.legend(loc='upper right', shadow=True, fontsize='x-large')
plt.show()