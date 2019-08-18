# print the hist of channels

import argparse
import numpy as np 
from matplotlib import pyplot as plt 

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--path', 
                    default='vgg_spl_1', type=str, help='path to txt file')
args = parser.parse_args()

data = []
fname = 'net_rgsm.txt'
file = open(fname,'r')
count = 0
wct = []
sz = []
tmp = 0
flag = 1
for line in file.readlines():
    if 'a' in line or 'm' in line or 'sz' in line:
        if 'sz' in line:
            curline = line[4:].replace(']',' ').replace(',','')
            curline = curline.strip().split(' ')
            szstr = [int(curline[0]),int(curline[1]),int(curline[2]),int(curline[3])]
            sz.append(szstr)

        if flag==0:
            wct.append(tmp)
            tmp = 0
        flag = 1
        pass
        # print(line)
    else:
        flag = 0
        line = line.replace('[',' ').replace(']',' ').replace(' ','\t')
        curline = line.strip().split('\t')
        # curline = curline.split(' ')
        # print(curline)
        for i in range(len(curline)):
            # print(curline[i])
            # print(float(curline[i]))
            if curline[i]!= '':
                data.append(float(curline[i]))
                if float(curline[i])<1e-15:
                    count += 1
                    tmp += 1
if flag==0:
    wct.append(tmp)
    tmp = 0

print(len(data))
file.close()
print(max(data),min(data))
print(type(data[0]))
print(count)

plt.xlim( [min(data)-0.003, 1+0.00001] )
plt.hist(data,500,color='C1')
plt.xlabel('channel norm')
plt.ylabel('count')
plt.title('RVSM on ResNet20')
# plt.title(fname.replace('.txt',''))
plt.text(0.6,0.9*30+20,'Number of zeros is: {:,.0f}'.format(count))
plt.show()

plt.xlim( [min(data), 2e-7] )
plt.hist(data, np.arange(0.,2e-7,1e-8),color='C1')
plt.xlabel('channel norm')
plt.ylabel('count')
plt.title('RVSM on ResNet20')
# plt.title(fname.replace('.txt',''))
plt.text(0.6*2e-7,0.9*30,'Number of zeros is: {:,.0f}'.format(count))
plt.show()


# print(sz)
adup1 = 0
adup2 = 0
ch_ad = 0
#fname = './weights/'+args.path+'/'+'count'
f_c = open(fname+'.txt', 'w')
for i in range(len(wct)):
    f_c.write('conv'+ str(i)+'	0 weights		total weights'+'\n')
    tmp = wct[i]
    np.set_printoptions(precision=9)
    tmp2 = np.prod(sz[i])
    tmp1 = tmp*tmp2/sz[i][1]
    adup1 += tmp1
    adup2 += tmp2
    ch_ad += sz[i][1]
    f_c.write(str(tmp)+'	'+str(tmp1)+'			'+str(tmp2)+'\n')
f_c.write(str(count)+'/'+str(ch_ad)+'\n')
f_c.write('channel sparsity(%): '+str(100.*count/ch_ad)+'\n')
f_c.write('weight sparsity(%): '+str(100.*adup1/adup2))

f_c.close()    