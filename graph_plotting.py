import numpy as np
from matplotlib import pyplot as plt
from math import sqrt

f1 = np.loadtxt("./data/quadrature_test/fejer_1")
f2 = np.loadtxt("./data/quadrature_test/fejer_2")
ga = np.loadtxt("./data/quadrature_test/gaussian")
le = np.loadtxt("./data/quadrature_test/legendre")
le_pr = np.loadtxt("./data/quadrature_test/legendre_proxy")
ra = np.loadtxt("./data/quadrature_test/radau")

fr_g_1 = np.loadtxt("./data/DI-entropy/frenkel_g_1")
fr_g_2 = np.loadtxt("./data/DI-entropy/frenkel_g_2")
fr_g_3 = np.loadtxt("./data/DI-entropy/frenkel_g_3")
fr_g_4 = np.loadtxt("./data/DI-entropy/frenkel_g_4")
fr_g_5 = np.loadtxt("./data/DI-entropy/frenkel_g_5")

fr_pbp_1 = np.loadtxt("./data/DI-entropy/frenkel_pbp_1")
fr_pbp_2 = np.loadtxt("./data/DI-entropy/frenkel_pbp_2")
fr_pbp_3 = np.loadtxt("./data/DI-entropy/frenkel_pbp_3")
fr_pbp_4 = np.loadtxt("./data/DI-entropy/frenkel_pbp_4")
fr_pbp_5 = np.loadtxt("./data/DI-entropy/frenkel_pbp_5")
fr_pbp_6 = np.loadtxt("./data/DI-entropy/frenkel_pbp_6")
fr_pbp_7 = np.loadtxt("./data/DI-entropy/frenkel_pbp_7")
fr_pbp_8 = np.loadtxt("./data/DI-entropy/frenkel_pbp_8")

fr_ra_2 = np.loadtxt("./data/DI-entropy/frenkel_radau_2")
fr_ra_4 = np.loadtxt("./data/DI-entropy/frenkel_radau_4")
fr_ra_6 = np.loadtxt("./data/DI-entropy/frenkel_radau_6")
fr_ra_8 = np.loadtxt("./data/DI-entropy/frenkel_radau_8")

fr_lin_2 = np.loadtxt("./data/DI-entropy/frenkel_linear_2")
fr_lin_3 = np.loadtxt("./data/DI-entropy/frenkel_linear_3")
fr_lin_4 = np.loadtxt("./data/DI-entropy/frenkel_linear_4")
fr_lin_5 = np.loadtxt("./data/DI-entropy/frenkel_linear_5")
fr_lin_6 = np.loadtxt("./data/DI-entropy/frenkel_linear_6")
fr_lin_7 = np.loadtxt("./data/DI-entropy/frenkel_linear_7")
fr_lin_8 = np.loadtxt("./data/DI-entropy/frenkel_linear_8")

ko_pbp_2 = np.loadtxt("./data/DI-entropy/kosaki_pbp_2")
ko_pbp_4 = np.loadtxt("./data/DI-entropy/kosaki_pbp_4")
ko_pbp_6 = np.loadtxt("./data/DI-entropy/kosaki_pbp_6")
ko_pbp_8 = np.loadtxt("./data/DI-entropy/kosaki_pbp_8")

ko_global_AB_2 = np.genfromtxt("./data/DI-entropy/CHSH/chsh_global_2M_full.csv", delimiter=',')
ko_global_AB_4 = np.genfromtxt("./data/DI-entropy/CHSH/chsh_global_4M_full.csv", delimiter=',')
ko_global_AB_6 = np.genfromtxt("./data/DI-entropy/CHSH/chsh_global_6M_full.csv", delimiter=',')
ko_global_AB_8 = np.genfromtxt("./data/DI-entropy/CHSH/chsh_global_8M_full.csv", delimiter=',')

fr_lin_AB_2 = np.loadtxt("./data/DI-entropy/CHSH/fr_lin_AB_2")
fr_lin_AB_4 = np.loadtxt("./data/DI-entropy/CHSH/fr_lin_AB_4")
fr_lin_AB_6 = np.loadtxt("./data/DI-entropy/CHSH/fr_lin_AB_6")
fr_lin_AB_8 = np.loadtxt("./data/DI-entropy/CHSH/fr_lin_AB_8")
fr_lin_AB_10 = np.loadtxt("./data/DI-entropy/CHSH/fr_lin_AB_10")

ref_AB = np.genfromtxt("./data/DI-entropy/CHSH/roger_global_chsh.csv", delimiter=',')

ref = np.loadtxt("./data/reference")

"""
plt.plot(fr_g_1[:,0], fr_g_1[:,1], label="order 1")
plt.plot(fr_g_2[:,0], fr_g_2[:,1], label="order 2")
plt.plot(fr_g_3[:,0], fr_g_3[:,1], label="order 3")
plt.plot(fr_g_4[:,0], fr_g_4[:,1], label="order 4")
plt.plot(fr_g_5[:,0], fr_g_5[:,1], label="order 5")
plt.plot(ref[:,0], ref[:,1], label="reference", linestyle="dashed")

plt.xlim(0.8,1.0)
plt.ylim(0, 1.1)
plt.legend()

plt.savefig("./data/DI-entropy/frenkel_global.png")


plt.plot(fr_pbp_1[:,0], fr_pbp_1[:,1], label="order 1")
plt.plot(fr_pbp_2[:,0], fr_pbp_2[:,1], label="order 2")
plt.plot(fr_pbp_3[:,0], fr_pbp_3[:,1], label="order 3")
plt.plot(fr_pbp_4[:,0], fr_pbp_4[:,1], label="order 4")
plt.plot(fr_pbp_5[:,0], fr_pbp_5[:,1], label="order 5")
plt.plot(fr_pbp_6[:,0], fr_pbp_6[:,1], label="order 6")
plt.plot(fr_pbp_7[:,0], fr_pbp_7[:,1], label="order 7")
plt.plot(fr_pbp_8[:,0], fr_pbp_8[:,1], label="order 8")
plt.plot(ref[:,0], ref[:,1], label="reference", linestyle="dashed")

plt.xlim(0.8,1.0)
plt.ylim(0, 1.1)
plt.legend()
plt.savefig("./data/DI-entropy/frenkel_point_by_point.png")

plt.plot(ko_pbp_2[:,0], ko_pbp_2[:,1], label="order 2")
plt.plot(ko_pbp_4[:,0], ko_pbp_4[:,1], label="order 4")
plt.plot(ko_pbp_6[:,0], ko_pbp_6[:,1], label="order 6")
plt.plot(ko_pbp_8[:,0], ko_pbp_8[:,1], label="order 8")
plt.plot(ref[:,0], ref[:,1], label="reference", linestyle="dashed")

plt.xlim(0.8,1.0)
plt.ylim(0, 1.1)
plt.legend()
plt.savefig("./data/DI-entropy/kosaki_point_by_point.png")

plt.plot(f1[:,0], f1[:,1], label="fejner_1")
plt.plot(f2[:,0], f2[:,1], label="fejner_2")
plt.plot(ga[:,0], ga[:,1], label="gaussian")
plt.plot(le[:,0], le[:,1], label="legendre")
plt.plot(le_pr[:,0], le_pr[:,1], label="legendre_proxy")
plt.plot(ra[:,0], ra[:,1], label="radau")

plt.legend()


plt.plot(fr_g_1[:,0], fr_g_1[:,1], label="frenkel global order 1")
plt.plot(fr_g_2[:,0], fr_g_2[:,1], label="frenkel global order 2")
plt.plot(ko_pbp_2[:,0], ko_pbp_2[:,1], label="kosaki point by point order 2")
plt.plot(ref[:,0], ref[:,1], label="reference", linestyle="dashed")

plt.xlim(0.8,1.0)
plt.ylim(0, 1.1)
plt.legend()
plt.savefig("./data/DI-entropy/comparison_2.png")



#plt.plot(fr_g_3[:,0], fr_g_3[:,1], label="frenkel global order 3")
plt.plot(fr_g_4[:,0], fr_g_4[:,1], label="frenkel order 4")
plt.plot(ko_pbp_4[:,0], ko_pbp_4[:,1], label="kosaki point by point order 4")
plt.plot(ref[:,0], ref[:,1], label="reference", linestyle="dashed")

plt.xlim(0.8,1.0)
plt.ylim(0, 1.1)
plt.legend()
plt.savefig("./data/DI-entropy/comparison_4.png")
"""

"""
plt.plot(fr_pbp_6[:,0], fr_pbp_6[:,1], label="frenkel point by point order 6")
plt.plot(ko_pbp_6[:,0], ko_pbp_6[:,1], label="kosaki point by point order 6")
plt.plot(ref[:,0], ref[:,1], label="reference", linestyle="dashed")

plt.xlim(0.8,1.0)
plt.ylim(0, 1.1)
plt.legend()
plt.savefig("./data/DI-entropy/comparison_6.png")

plt.plot(fr_pbp_8[:,0], fr_pbp_8[:,1], label="frenkel point by point")
plt.plot(ko_pbp_8[:,0], ko_g_8[:,1], label="kosaki point by point")

plt.xlim(0.8,1.0)
plt.ylim(0, 1.1)
plt.legend()
plt.savefig("./data/DI-entropy/comparison_8.png")

plt.xlim(2,20)
plt.ylim(0,60)

plt.savefig("./data/quadrature_test/comparison.png")


plt.plot(fr_ra_2[:,0], fr_ra_2[:,1], label = "order 2")
plt.plot(fr_ra_4[:,0], fr_ra_4[:,1], label = "order 4")
plt.plot(fr_ra_6[:,0], fr_ra_6[:,1], label = "order 6")
plt.plot(fr_ra_8[:,0], fr_ra_8[:,1], label = "order 8")
plt.plot(ref[:,0], ref[:,1], label = "reference", linestyle="dashed")

plt.xlim(0.8, 1.0)
plt.ylim(0.0, 1.0)
plt.legend()

plt.savefig("./data/frenkel_radau.png")


plt.plot(fr_lin_2[:,0], fr_lin_2[:,1], label = "order 2")
#plt.plot(fr_lin_2[:,0], fr_lin_3[:,1], label = "order 3")
plt.plot(fr_lin_4[:,0], fr_lin_4[:,1], label = "order 4")
#plt.plot(fr_lin_5[:,0], fr_lin_5[:,1], label = "order 5")
plt.plot(fr_lin_6[:,0], fr_lin_6[:,1], label = "order 6")
#plt.plot(fr_lin_7[:,0], fr_lin_7[:,1], label = "order 7")
plt.plot(fr_lin_8[:,0], fr_lin_8[:,1], label = "order 8")
plt.plot(ref[:,0], ref[:,1], label = "reference", linestyle="dashed")

plt.xlim(0.8, 1.0)
plt.ylim(0.0, 1.0)
plt.legend()

plt.savefig("./data/frenkel_linear_even.png")


plt.plot(fr_lin_8[:,0], fr_lin_8[:,1], label = "frenkel linear")
plt.plot(ko_pbp_8[:,0], ko_pbp_8[:,1], label = "kosaki radau")
plt.plot(ref[:,0], ref[:,1], label = "reference", linestyle="dashed")

plt.xlim(0.8, 1.0)
plt.ylim(0.0, 1.0)
plt.legend()

plt.savefig("./data/fr_lin_ko_ra_comparison_8.png")


trm50 = [np.loadtxt('./data/TraceMinus/trm_'+str(i)+'_50') for i in range(82,101)]
trm100 = [np.loadtxt('./data/TraceMinus/trm_'+str(i)+'_100') for i in range(82,101)]
trmd100 = [np.loadtxt('./data/TraceMinus/trmd_'+str(i)+'_100') for i in range(82,101)]

for i in range(19):
    plt.plot(trmd100[i][:,0], trmd100[i][:,1])
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.savefig('./data/TraceMinus/trmd100_'+str(i+82)+".png")
    plt.close()


plt.plot(ko_global_AB_2[0], ko_global_AB_2[1], label="M = 2")
plt.plot(ko_global_AB_4[0], ko_global_AB_4[1], label="M = 4")
plt.plot(ko_global_AB_6[0], ko_global_AB_6[1], label="M = 6")
plt.plot(ko_global_AB_8[0], ko_global_AB_8[1], label="M = 8")

plt.plot(ref_AB[0], ref_AB[1], label = "Upper bound", linestyle = "dashed")

plt.xlim(0.75, (sqrt(2)+2)/4)
plt.ylim(0, 1.6)
plt.legend()

plt.savefig("./data/DI-entropy/CHSH/kosaki_AB.png")
"""

plt.plot(fr_lin_AB_2[:,0], fr_lin_AB_2[:,1], label = "M = 2")
plt.plot(fr_lin_AB_4[:,0], fr_lin_AB_4[:,1], label = "M = 4")
plt.plot(fr_lin_AB_6[:,0], fr_lin_AB_6[:,1], label = "M = 6")
plt.plot(fr_lin_AB_8[:,0], fr_lin_AB_8[:,1], label = "M = 8")
plt.plot(fr_lin_AB_10[:,0], fr_lin_AB_10[:,1], label = "M = 10")

plt.plot(ref_AB[0], ref_AB[1], label = "Upper bound", linestyle = "dashed")

plt.xlim(0.75, (sqrt(2)+2)/4)
plt.ylim(0, 1.6)
plt.legend()

plt.savefig("./data/DI-entropy/CHSH/frenkel_AB.png")
