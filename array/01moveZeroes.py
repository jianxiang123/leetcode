class Solution(object):
    def moveZeroes1(self, nums):
        # 暴力法 time O(n^2)
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                if nums[i] !=0:
                    nums[i],nums[j]=nums[j],nums[i]
    def moveZeroes2(self,nums):
        #index time O(n)
        j=0
        for i in range(len(nums)):
            if nums[i] !=0:
                nums[i], nums[j] = nums[j], nums[i]
                j+=1
    def moveZeroes3(self,nums):
        # in-place
        j=0
        for i in range(len(nums)):
            if nums[i] !=0:
                nums[j]=nums[i]
                j+=1
        while j<len(nums):
            nums[j]=0
            j+=1
    def moveZeroes4(self,nums):
        # 库函数
        nums.sort(key=lambda x:x==0)
