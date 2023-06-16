import pickle as pkl
import numpy as np

# packed_fp: array(n, 256), values: tensor(n, 1)
# after unpack: array(n, 2048)
ground_truth = pkl.load(open("data/Multi_Step_Task/target_mol_route.pkl", "rb"))

found = pkl.load(open('retro_star/retro_star/results/plan.pkl', "rb"))

route_ = found['routes']
route = []
for r in route_:
    if r:
        reactions_ = r.serialize().split('|')
        reactions = []
        for r in reactions_:
            r = r.split('>')
            r = '>>'.join([r[0],r[2]])
            reactions.append(r)
        route.append(reactions)
    else:
        route.append(None)

result = []
for i in range(len(route)):
    if route[i]:
        set_gt = set(ground_truth[i])
        set_route = set(route[i])
        common_elements = set_gt & set_route
        acc = len(common_elements)/float(len(set_gt))
        result.append((len(set_gt),len(set_route),acc))
    else:
        acc = 0.0
        result.append((len(set_gt),0,acc))

plot_list = []
for i,r in enumerate(result):
    print(f"Mol {i+1}:    Groundtruth reactions:{r[0]}    Retro* found reactions:{r[1]}   Accuracy:{r[2]}")
    if r[1]:
        plot_list.append(r[2])


import matplotlib.pyplot as plt

plt.cla()
y = np.array(plot_list)
print(sum(y==1))
print(np.mean(y))
x = np.arange(y.shape[0]) + 1

plt.plot(x,y,label='')
plt.legend()
plt.title('Overlap ratio between ground-truth route and Retro* route')
plt.xlabel('mol id')
plt.ylabel('overlap ratio')
plt.savefig('fig.png')