# Simpaire
This repository contains all the python files and the setup needed te reproduce the biomechanical simulations of a decrease hip extension during stance and a decrease knee flexion during swing

## Folder content
+ **SCONE_files**:This repository contains SCONE files designed for simulating two distinct gait impairments:
Markup : * Decrease Knee Flexion during the Swing Phase
          * Maximum Knee Angle = 70 deg
          * Maximum Knee Angle = 68 deg
          * Maximum Knee Angle = 67 deg
          * Maximum Knee Angle = 66.5 deg
          * Maximum Knee Angle = 66 deg
          * Maximum Knee Angle = 65.5 deg
          * Maximum Knee Angle = 65 deg
      * Decrease Hip Extension during the Stance Phase
          * Minimum Hip Angle = -15 deg
          * Minimum Hip Angle = -13 deg
          * Minimum Hip Angle = -11 deg
          * Minimum Hip Angle = -9 deg
          * Minimum Hip Angle = -7 deg
          * Minimum Hip Angle = -5 deg 
          * Minimum Hip Angle = -3 deg
          * Minimum Hip Angle = 0 deg

These files are intended for use with the SCONE software to simulate gait patterns with varying degrees of knee flexion during the swing phase and hip extension during the stance phase.

+ **Python_files**: This respository contains a python file that allows to visualize the output of the simulations (healthy_impaired.py) and a python file that computes the keppler mapper on a 112x7 matrix (KM_high_dim_new_standardization_without_grf_swing.py).
