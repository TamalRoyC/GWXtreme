from GWXtreme.eos_model_selection import Model_selection as modsel
import numpy as np
import h5py

#EoS models to compare

EoS1="APR4_EPP"
EoS2="SLY"

# Posterior File generated using Phenom-PNRT waveform. For examle, download 170817's posterior file from from https://dcc.ligo.org/public/0152/P1800115/012/EoS-insensitive_posterior_samples.dat
filename='/home/anarya.ray/gwxtreme-project/repos/3d-kde/EoS-insensitive_posterior_samples.dat'

#parse file and save in a format compatible with Model_selection input

data = np.recfromtxt(filename, names=True)

m1,m2,l1,l2=data['m1_source_frame_Msun'],data['m2_source_frame_Msun'],data['Lambda1'],data['Lambda2']
q=m2/m1
mc=(m1*m2)**(3./5.)/(m1+m2)**(1./5.)
np.savetxt("EoS-insensitive_posterior_samples.dat",np.c_[mc,q,m1,m2,l1,l2],header="mc_source \t q \t m1_source \t m2_source \t lambda1 \t lambda2",delimiter="\t",fmt="%f")

#initialize GWXtrme.eos_model_selection.Model_Selection object for 3 dimensional KDE
model3d=modsel("EoS-insensitive_posterior_samples.dat",kdedim=3,Ns=len(l1))

#Print BAyesfactor, compare with table B1 column 2 of https://dcc.ligo.org/public/0157/P1800379/010/ns_model_comparison_P1800379_v10.pdf. find the values of EoS1 and EoS2 rows from that column and take their ratio.comare with:
print(EoS1+"/"+EoS2+" Bayesfactor for PhenomPNRT, using GWXtreme:",model3d.computeEvidenceRatio(EoS1,EoS2))
