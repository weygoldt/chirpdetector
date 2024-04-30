# Refactoring tasks 📋

Get yourself a cup of coffee and happy refactoring! 🚀

## chirpdetector/detection/detection_models.py

- [ ] Refactor the ABC to always get an input transform and an output transform
      (chirpdetector/detection/detection_models.py:10)

## chirpdetector/detection/detect_chirps.py

- [ ] Move shape to config (chirpdetector/detection/detect_chirps.py:483)
- [ ] Create a function to extract chirp spec snippets, generate a chirp
      dataset, and save it to disk with h5py
      (chirpdetector/detection/detect_chirps.py:502)

## chirpdetector/.trash/detect_chirps.py

- [ ] Improve efficiency in finding a new hop length that is a multiple of the
      chunksize but close to the old hop length
      (chirpdetector/.trash/detect_chirps.py:619)
- [ ] Improve efficiency in finding a new hop length that is a multiple of the
      chunksize (chirpdetector/.trash/detect_chirps.py:640)
- [ ] Make the frequency limits for the spectrogram consistent
      (chirpdetector/.trash/detect_chirps.py:659)
- [ ] Implement the same checks as for small chunks to handle duplicates in the
      time array (chirpdetector/.trash/detect_chirps.py:916)

## chirpdetector/.trash/convert_data_old.py

- [ ] Improve the subset naming convention in gridtools
      (chirpdetector/.trash/convert_data_old.py:309)
- [ ] Remove this when gridtools is fixed
      (chirpdetector/.trash/convert_data_old.py:423)

## chirpdetector/.trash/assign_chirps.py

- [ ] Update docstrings in this module
      (chirpdetector/.trash/assign_chirps.py:30)
- [ ] Update the cost function to prioritize higher values for better choices
      instead of lower values (chirpdetector/.trash/assign_chirps.py:499)
- [ ] Save envs to disk for plotting
      (chirpdetector/.trash/assign_chirps.py:545)

## chirpdetector/datahandling/output_data_model.py

- [ ] Implement the `_check_health` method in the `ChirpDatasetSaver` class
      (chirpdetector/datahandling/output_data_model.py:143)

## chirpdetector/datahandling/bbox_tools.py

- [ ] Check and fix the issue with the weights calculation for interpolation
      (chirpdetector/datahandling/bbox_tools.py:86)
