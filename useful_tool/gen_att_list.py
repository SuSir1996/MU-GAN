# -*- coding: utf-8 -*-
import numpy as np
x = np.array([['a',-1,2,3],['b',4,-5,-6]])
#np.savetxt('test.txt', x) 
np.savetxt('test1.txt', x,fmt='%s') 
np.savetxt('test2.txt', x, delimiter=',') 
np.savetxt('test3.txt', x,newline='a') 
np.savetxt('test4.txt', x,delimiter=',',newline='a') 
np.savetxt('test5.txt', x,delimiter=',',header='abc') 
np.savetxt('test6.txt', x,delimiter=',',footer='abc')
np.savetxt('test6.txt', x,delimiter=' ',footer='abc')
