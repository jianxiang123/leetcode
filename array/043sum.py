def threeSum(nums):
    # 暴力
    res=[]
    for i in range(len(nums)):
        for j in range(i+1,len(nums)):
            for k in range(j+1,len(nums)):
                if nums[i]+nums[j]+nums[k]==0:
                    res.append([nums[i],nums[j],nums[k]])
    return res
def threeSum1(nums):
    #排序 + 双指针法
    nums.sort()
    n=len(nums)
    res=[]
    for i in range(n):
        if nums[i]>0:
            return res
        if nums is None or len(nums)<3:
            return res
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        l=i+1
        r=n-1
        while l<r:
            sum=nums[i]+nums[l]+nums[r]
            if sum > 0:
                l+=1
                continue
            if sum< 0:
                r-=1
                continue
            if sum == 0:
                res.append([nums[i],nums[l],nums[r]])
                while l<r and nums[l]==nums[l+1]:
                    l+=1
                while l<r and nums[r]==nums[r-1]:
                    r-=1
                l+=1
                r-=1
    return res