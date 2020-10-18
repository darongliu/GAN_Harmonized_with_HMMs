import matplotlib.pyplot as plt

# X axis
cluster_num = [50,100,200,300,500,800,1000]

# upper bound
upper_bound = [48.18,42.86,38.11,36.21,34.12,32.505,31.745]
# matched gumbel performance
match_gumbel = [54.07,47.61,41.53,40.2,38.205,36.59,35.925]
# match w/o tokenization performance
match_without_token_per = 27.93
# nonmatched gumbel performance
non_match_gumbel = [50.24,48.15,45.585,43.4,44.54,45.205,45.87]
# nonmatch w/o tokenization performance
non_match_without_token_per = 34.4

assert (len(cluster_num) == len(upper_bound) == len(match_gumbel) == len(non_match_gumbel))

# set figure size
plt.figure(figsize=(30, 10),dpi=100,linewidth = 2)
#fig , ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(30, 10),dpi=100,linewidth = 2)

######### plot match #########
ax1=plt.subplot(121)

# set figure size
#plt.figure(figsize=(15, 10),dpi=100,linewidth = 2)

# draw a line, mark shape(s-)
plt.plot(cluster_num,upper_bound,'s-',color = 'r', label="purity")
# draw a line, mark shape(o-)
plt.plot(cluster_num,match_gumbel,'o-',color = 'g', label="w/ tokenization")
# draw horiz line 
plt.axhline(y=match_without_token_per, color='r', linestyle='--', label='w/o tokenization')

# set title and the distance to x and y
plt.title("(a) match", x=0.028, y=1.03, fontsize=25)

# set font size
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# label the name of x axis (labelpad denote the distance to the figure)
plt.xlabel("Cluster number", fontsize=25, labelpad = 15)
# label the name of y axis (labelpad denote the distance to the figure)
plt.ylabel("Phoneme error rate", fontsize=25, labelpad = 20)

# the name of line
plt.legend(loc="best", fontsize=20)

######### plot nonmatch #########
plt.subplot(122, sharex=ax1, sharey=ax1)

# set figure size
#plt.figure(figsize=(15, 10),dpi=100,linewidth = 2)

# draw a line, mark shape(s-)
plt.plot(cluster_num,upper_bound,'s-',color = 'r', label="purity")
# draw a line, mark shape(o-)
plt.plot(cluster_num,non_match_gumbel,'o-',color = 'g', label="w/ tokenization")
# draw horiz line 
plt.axhline(y=non_match_without_token_per, color='r', linestyle='--', label='w/o tokenization')

# set title and the distance to x and y
plt.title("(b) nonmatch", x=0.05, y=1.03, fontsize=25)

# set font size
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# label the name of x axis (labelpad denote the distance to the figure)
plt.xlabel("Cluster number", fontsize=25, labelpad = 15)
# label the name of y axis (labelpad denote the distance to the figure)
plt.ylabel("Phoneme error rate", fontsize=25, labelpad = 20)

# the name of line
plt.legend(loc="best", fontsize=20)

######### output #########
# output the figure
plt.tight_layout()
plt.savefig('/Users/liudarong/Desktop/result.png')
#plt.show()
