class Solution(object):
    def twoSum(self, nums, target):
        # 暴力 time O(n^2)
        res=[]
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                if nums[i]+nums[j]==target:
                    res.append(i)
                    res.append(j)
        return res
    def twoSum1(self,nums,target):
        # 哈希法 time O(n)
        dic={}
        for i in range(len(nums)):
            if nums[i] in dic:
                return [i,dic[nums[i]]]
            else:
                dic[target - nums[i]]=i


