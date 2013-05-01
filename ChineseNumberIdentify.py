#-*- coding:utf-8 -*-
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer

import numpy as np
import os,Image
import cPickle
import datetime

def get_train_samples(input_num,output_num):
    '''
    从new_samples文件夹中读图，根据输入数和输出数制作样本，每一原始样本加入随机噪音生成100个样本
    '''
    print 'getsample start.'
    sam_path='./new_samples'
    samples = SupervisedDataSet(input_num,output_num)
    nlist = os.listdir(sam_path)
    t=int(np.sqrt(input_num))
    for n in nlist:
        file = os.path.join(sam_path,n)
        im = Image.open(file)
        im = im.convert('L')
        im = im.resize((t,t),Image.BILINEAR)
        #im.point(lambda x:255 if x==255 else 0)
        #im = im.convert('1')
        #im.save('./buf/'+n)
        buf = np.array(im).reshape(input_num,1)
        buf = buf<200
        buf = tuple(buf)
        buf1=int(n.split('.')[0])
        buf2=range(output_num)
        for i in range(len(buf2)):
            buf2[i] = 0
        buf2[buf1]=1
        buf2 = tuple(buf2)
        samples.addSample(buf,buf2)
        for i in range(100):
            # if not i%100:
                # print n,'  ',i
            buf3 = list(buf)
            for j in range(len(buf)/20):
                buf3[np.random.randint(len(buf))] = bool(np.random.randint(2))
            samples.addSample(tuple(buf3),buf2)
    return samples 

def get_test_samples(input_num):
    '''
    从new_test文件夹读取测试数据
    '''
    print 'Get test samples start.'
    test_path='./new_test'
    samples = SupervisedDataSet(input_num,1)
    nlist = os.listdir(test_path)
    t=int(np.sqrt(input_num))
    for n in nlist:
        file = os.path.join(test_path,n)
        im = Image.open(file)
        im = im.convert('L')
        im = im.resize((t,t),Image.BILINEAR)
        #im.point(lambda x:255 if x==255 else 0)
        #im = im.convert('1')
        buf = np.array(im).reshape(input_num,1)
        buf = buf<200
        buf = tuple(buf)
        samples.addSample(buf,1)
    return samples
        


    
class net:
    '''
    网络的定义
    '''
    def __init__(self,input_num,hide_node_num,output_num):
        '''
        根据参数初始化网络
        '''
        self.input_num = input_num
        self.hide_node_num = hide_node_num
        self.output_num = output_num
        self.network = buildNetwork(input_num,hide_node_num,output_num,bias=True)
        
    def train(self,samples,epsilon):
        '''
        训练函数
        '''
        print 'Train start.'
        trainer = BackpropTrainer(self.network,samples)
        e = 100
        n=0
        #e=trainer.trainUntilConvergence()
        while e>epsilon:
            e=trainer.train()
            n+=1
            #if not n%100:
            print n,' done,e=',e
            if not n%10:
                self.save()
            if n>=100:break
        self.save() 
        print 'Train end.'
        return e
    
    def run(self,samples):
        '''
        测试
        '''
        print 'Test start.'
        result = []
        for sample in samples['input']:
            buf = self.network.activate(sample)
            buf= list(buf)
            result.append(buf.index(max(buf)))
        print 'Result ',result
        result_path = './results/'
        filename = str(self.input_num)+'-'+str(self.hide_node_num)+'-'+str(self.output_num)+'new.txt'
        with open(result_path+filename,'w') as f:
            result = str(result)
            f.write(result)
    
    def save(self):
        '''
        保存训练好的网络
        '''
        print 'saving'
        save_path = './save/'
        filename = str(self.input_num)+'-'+str(self.hide_node_num)+'-'+str(self.output_num)+'new.cPickle'
        with open(save_path+filename,'wb') as f:
            cPickle.dump(self.network,f)
        print 'done'
            
    def load(self):
        '''
        从存档中加载训练好的网络
        '''
        print 'loading'
        save_path ='./save/'
        filename = str(self.input_num)+'-'+str(self.hide_node_num)+'-'+str(self.output_num)+'new.cPickle'
        if filename in os.listdir('./save/'):
            with open(save_path+filename,'rb') as f:
                self.network = cPickle.load(f)
        print 'done'
        
        
def main():
    start=datetime.datetime.now()
    output_num = 10
    epsilon = 0.01
    input_num=20*20
    hide_node_num = 100
    filename = str(input_num)+'-'+str(hide_node_num)+'-'+str(output_num)+'new.cPickle'
    net1 = net(input_num,hide_node_num,output_num)
    if filename in os.listdir('./save/'):
        net1.load()
    else:
        samples = get_train_samples(input_num,output_num)
        net1.train(samples,epsilon)
        net1.save()
    net1.run(get_test_samples(input_num))
    end =datetime.datetime.now()
    print 'Time ',end-start
    
main()
        