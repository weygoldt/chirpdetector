# Data structure

To detect chirps on a recording of single or multiple electrodes, the dataset
must fulfill the following requirements:

- The dataset is a single directory containing subdirectories for each single
  `.raw` or `.wav` file, i.e. for each recording.
- Each raw recording file, i.e. the `.raw` or `.wav` file is named `traces_grid1.{file_ext}`.
- The fundamental frequencies of the fish in the recordings must be already
  tracked using the [wavetracker](https://github.com/tillraab/wavetracker) project.

An example directory structure could look like the following:

```
 dataset_root
├──  chirpdetector.toml    # the chirpdetector config
├──  2019-11-25-09_59      # dir of first recording
│  ├──  fund_v.npy         # wavetracker
│  ├──  ident_v.npy        # wavetracker
│  ├──  idx_v.npy          # wavetracker
│  ├──  sign_v.npy         # wavetracker
│  ├──  times.npy          # wavetracker
│  └──  traces_grid1.wav   # raw recording
├──  2019-11-26-09_54      # dir of second recording
│  ├──  fund_v.npy         # ...
...
```

If these requirements are met, detecting chirps should be no issue.
