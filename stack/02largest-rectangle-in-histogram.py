class Solution(object):
    def largestRectangleArea(self, heights): #暴力 time O(n^2)
        maxArea=0
        n=len(heights)
        for i in range(n):
            left_i=i
            right_i=i
            while left_i>0 and heights[left_i]>=heights[i]:
                left_i-=1
            while right_i<n and heights[right_i]>=heights[i]:
                right_i+=1
            maxArea=max(maxArea,(right_i-left_i-1)*heights[i])
        return maxArea

    def largestRectangleArea1(self, heights):  # 栈 time O(n)
        heights.append(0)  # very important!!
        stack=[-1]
        ans=0
        for i in range(len(heights)):
            while heights[i]<heights[stack[-1]]:
                h=heights[stack.pop()]
                w=i-stack[-1]-1
                ans=max(ans,h*w)
            stack.append(i)
        heights.pop()
        return ans