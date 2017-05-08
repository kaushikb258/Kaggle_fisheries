import numpy as np

f = open("fish_output", "r") 

val_bb_acc = []
val_class_acc = []

for line in f:
 if 'val_loss' in line:
   a = line.split('val_bb_acc:')[1] 
   c = a.split('-')[0]
   val_bb_acc.append(float(c))
   b = a.split('val_class_acc:')[1]
   val_class_acc.append(float(b))


val_bb_acc = np.array(val_bb_acc)
val_class_acc = np.array(val_class_acc)



k = []
for i in range(val_bb_acc.shape[0]):
 k.append([i, val_bb_acc[i], val_class_acc[i]])

np.savetxt('my_accuracy',np.array(k),fmt='%i,%.4f,%.4f') 
