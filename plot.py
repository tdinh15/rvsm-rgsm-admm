# print the hist of channels

import argparse
import numpy as np 
from matplotlib import pyplot as plt 

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--path', 
                    default='vgg_spl_1', type=str, help='path to txt file')
args = parser.parse_args()

data1 = []
data2 = []
fname1 = 'net1.txt'
fname2 = 'net2.txt'
file = open(fname1,'r')
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
                data1.append(float(curline[i]))
                if float(curline[i])<1e-15:
                    count += 1
                    tmp += 1
if flag==0:
    wct.append(tmp)
    tmp = 0
    
file2 = open(fname2,'r')    
for line in file2.readlines():
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
                data2.append(float(curline[i]))
                if float(curline[i])<1e-15:
                    count += 1
                    tmp += 1
if flag==0:
    wct.append(tmp)
    tmp = 0

print(len(data1))
file.close()
print(max(data1),min(data1))
print(type(data1[0]))
print(count)

plt.xlim( [-0.002, 1+0.00001] )
plt.hist(data1,500)
plt.hist(data2,500)
plt.xlabel('channel norm')
plt.ylabel('count')
plt.title(fname1.replace('.txt',''))
plt.text(0.6,0.9*count+20,'Number of zeros is: {:,.0f}'.format(count))
plt.show()

plt.xlim( [min(data1), 2e-7] )
plt.hist(data1, np.arange(0.,2e-7,1e-8))
plt.hist(data2, np.arange(0.,2e-7,1e-8))
plt.xlabel('channel norm')
plt.ylabel('count')
plt.title(fname1.replace('.txt',''))
plt.text(0.6*2e-7,0.9*count,'Number of zeros is: {:,.0f}'.format(count))
plt.show()




# # print(sz)
# adup1 = 0
# adup2 = 0
# ch_ad = 0
# #fname = './weights/'+args.path+'/'+'count'
# f_c = open(fname+'.txt', 'w')
# for i in range(len(wct)):
    # f_c.write('conv'+ str(i)+'	0 weights		total weights'+'\n')
    # tmp = wct[i]
    # np.set_printoptions(precision=9)
    # tmp2 = np.prod(sz[i])
    # tmp1 = tmp*tmp2/sz[i][1]
    # adup1 += tmp1
    # adup2 += tmp2
    # ch_ad += sz[i][1]
    # f_c.write(str(tmp)+'	'+str(tmp1)+'			'+str(tmp2)+'\n')
# f_c.write(str(count)+'/'+str(ch_ad)+'\n')
# f_c.write('channel sparsity(%): '+str(100.*count/ch_ad)+'\n')
# f_c.write('weight sparsity(%): '+str(100.*adup1/adup2))
# f_c.close()    
