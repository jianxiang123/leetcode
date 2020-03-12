class Solution(object):
    def climbStairs1(self, n):
        # recursion
        if n ==1:
            return 1
        if n ==2:
            return 2
        return self.climbStairs1(n-1)+self.climbStairs1(n-2)
    def climbStairs2(self,n):
        #斐波那契数列
        if n ==1:
            return 1
        res=[0 for i in range(n)]
        res[0],res[1]=1,2
        for i in range(2,n):
            res[i]=res[i-1]+res[i-2]
        return res[-1]
    def climbStairs3(self,n):
        #iteration
        if n ==1:
            return 1
        a,b=1,2
        for i in range(2,n):
            tem=b
            b=a+b
            a=tem
        return b
    def climbStairs4(self,n):
        #memorization (list)
        if n ==1:
            return 1
        lis=[-1 for i in range(n)]
        lis[0],lis[1]=1,2
        return self.help(n-1,lis)
    def help(self,n,lis):
        if lis[n]<0:
            lis[n]=self.help(n-1,lis)+self.help(n-2,lis)
        return lis[n]
    # memorization (dictionary)
    def __init__(self):
        self.dic={1:1,2:2}
    def climbStairs5(self,n):
        if n not in self.dic:
            self.dic[n]=self.climbStairs5(n-1)+self.climbStairs5(n-2)
        return self.dic[n]