class Solution(object):
    def maxArea1(self, height):
        # 暴力 time O(n^2)
        maxWater=0
        for i in range(len(height)):
            for j in range(i+1,len(height)):
                high=min(height[i],height[j])
                maxWater=max(maxWater,(j-i)*high)
        return maxWater
    def maxArea2(self,height):
        # 双指针法 time O(n)
        left,right,maxWater,minHigh=0,len(height)-1,0,0
        while left<right:
            if height[left]<height[right]:
                minHigh,left=height[left],left+1
            else:
                minHigh,right=height[right],right-1
            maxWater=max(maxWater,(right-left+1)*minHigh)
        return maxWater