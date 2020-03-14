class Solution(object):
    def maxSlidingWindow(self, nums, k): #暴力
        n=len(nums)
        if n*k==0:
            return []
        return [max(nums[i:i+k])for i in range(n-k+1)]
    def maxSlidingWindow1(self, nums, k): #队列
        ans=[]
        queue=[]
        for i,v in enumerate(nums):
            if queue and queue[0]<i-k:
                queue=queue[1:]
            while queue and nums[queue[-1]]<v:
                queue.pop()
            queue.append(i)
            if i+1>=k:
                ans.append(nums[queue[0]])
        return ans
