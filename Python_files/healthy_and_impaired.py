#This script generates:
# - a plot depicting the average signal for the impaired gait
# - a plot depicting the average signal for the healthy slow gait
# - a plot in which both the average signal for the impaired gait and
# the average signal for the healthy slow gait are depicted
# - calculate and export the relative difference mean_impaired - mean_healthy
# for further analysis with the keppler mapper


from utils import *

#state_file = os.path.abspath( "C:/Users/lea.chabrowski/Documents/SCONE/results/Right_hip_red_max_minus_0_deg_6_corrected_measure.f0914m.GH2010v8.S05W.D15.I/0425_8.362_0.779.par.sto")

state_file = os.path.abspath( "C:/Users/lea.chabrowski/Documents/SCONE/results/Right_hip_red_max_minus_3_deg_9.f0914m.GH2010v8.S05W.D15.I/0347_0.702_0.697.par.sto")

#state_file = os.path.abspath( "C:/Users/lea.chabrowski/Documents/SCONE/results/Right_hip_red_max_minus_5_deg_4.f0914m.GH2010v8.S05W.D15.I (1)/0443_0.671_0.669.par.sto")

#state_file = os.path.abspath( "C:/Users/lea.chabrowski/Documents/SCONE/results/Right_hip_red_max_minus_7_deg_6.f0914m.GH2010v8.S05W.D15.I/0401_0.646_0.639.par.sto")


#state_file = os.path.abspath( "C:/Users/lea.chabrowski/Documents/SCONE/results/Right_hip_red_max_minus_9_deg_4.f0914m.GH2010v8.S05W.D15.I/0334_50.362_0.647.par.sto")


#state_file = os.path.abspath( "C:/Users/lea.chabrowski/Documents/SCONE/results/Right_hip_red_max_minus_11_deg_5.f0914m.GH2010v8.S05W.D15.I/0277_0.630_0.626.par.sto")


#state_file = os.path.abspath( "C:/Users/lea.chabrowski/Documents/SCONE/results/Right_hip_red_max_minus_13_deg_4.f0914m.GH2010v8.S05W.D15.I/0344_0.626_0.620.par.sto")

#state_file = os.path.abspath( "C:/Users/lea.chabrowski/Documents/SCONE/results/Right_knee_red_max_50_deg_min_BothKnee_0_deg_18.f0914m.GH2010v8.S05W.D15.I/0212_42.115_2.749.par.sto")

#healthy slow gait (run with optimized initial guess ALice's code)
state_file_healthy = os.path.abspath( "C:/Users/lea.chabrowski/Documents/SCONE/results/max_isometric_force_90.f0914m.GH2010v8.S05W.D15.I/0143_0.595_0.589.par.sto")



output_file = "C:\\Users\\lea.chabrowski\\Desktop"
state_healthy = read_from_storage(state_file_healthy)
side = 'l'

state = read_from_storage(state_file)

#Generate a plot depicting the average signal for the
# healthy slow gait scenario.
plot_scone_joint_mean_kinematics(state_healthy, side, output_file=output_file)
plt.show()

#Generate a plot depicting the average signal for the impaired gait
plot_scone_joint_mean_kinematics(state, side, output_file=output_file)
plt.show()

#Generate a plot in which both the average signal for the impaired gait and
# the average signal for the healthy slow gait are depicted
plot_scone_joint_mean_kinematics_healthy_and_impaired(state_healthy, state, side, output_file=None)
plt.legend()
plt.legend(fontsize=8)
plt.show()

#Return a matrix composed of 55 columns because for each parameters
#(ankle/knee/hip joint angle, ankle/knee/hip moment, pelvis tilt, ground reaction force)
# the average, minimum, maximum and standard deviation per stance and per swing was computed.
output_healthy = output_matrix_kinematics(state_healthy, side)
output_impaired = output_matrix_kinematics(state, side)

output_healthy_mean = output_healthy.mean(axis = 0)
output_impaired_mean = output_impaired.mean(axis = 0)

# Calculate the relative difference (impaired - healthy)
relative_difference_mean = output_healthy_mean - output_impaired_mean
relative_difference_mean.to_csv("C:\\Users\\lea.chabrowski\\Desktop\\new_standardization\\relative_difference_Right_hip_red_max_minus_13_deg_4_corrected_measure_r.csv")
