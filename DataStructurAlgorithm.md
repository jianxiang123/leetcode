# array

## 01 move-zeroes

### python



```python
# in-place
def moveZeroes(self, nums):
    zero = 0  # records the position of "0"
    for i in xrange(len(nums)):
        if nums[i] != 0:
            nums[i], nums[zero] = nums[zero], nums[i]
            zero += 1
```



```python
def moveZeroes(self, nums):
	for i in range(len(nums))[::-1]:
        if nums[i] == 0:
            nums.pop(i)
            nums.append(0)
```



```python
def moveZeroes(nums):
    j=0
    for i in range(len(nums)):
       if nums[i] !=0:
           nums[j]=nums[i]
           j+=1
    while j<len(nums):
        nums[j]=0
        j+=1
```



```python
def moveZeroes(nums):
    nums.sort(key=lambda x: x == 0)
```

### Go



```go
func moveZeroes(nums []int) {

	index := 0

	for i := 0; i < len(nums); i++ {
		if nums[i] != 0 {
			nums[index] = nums[i]
			index++
		}
	}
	for i := index; i < len(nums); i++ {
		nums[i] = 0
	}
}
```



```go
func moveZeroes(nums []int)  {
    b := nums[:0]
	lend := 0
	for _, x := range nums {
		if x!=0 {
			b = append(b, x)	
		} else {
			lend++
		}	
	}
	for lend > 0 {
		b = append(b, 0)
		lend--
	}
}
```



```go
func moveZeroes(nums []int) {
	for i, j := 0, 0; j < len(nums); {
		for ; i < len(nums) && nums[i] != 0; i++ {}
		for ; j < len(nums) && (i >= j || nums[j] == 0); j++ {}
		if j < len(nums) {
			nums[i], nums[j] = nums[j], nums[i]
		}
	}
}
```



```go
func moveZeroes(nums []int)  {

	slow,fast:=0,0
	for fast<len(nums){
		if nums[fast]!=0 {
			nums[slow],nums[fast]=nums[fast],nums[slow]
			slow++
		}
		fast++
	}
}
```



## 02 container-with-most-water

### python



```python
def maxArea(self, height):
    left, right, maxWater, minHigh = 0, len(height) - 1, 0, 0
    while left < right:
        if height[left] < height[right]:
            minHigh, left = height[left], left+1
        else:
            minHigh,right = height[right], right-1
        maxWater = max(maxWater, (right - left + 1) * minHigh)
    return maxWater
```



### go

```python
func maxArea(height []int) int {
    mx := 0
    i, j := 0, len(height) - 1
    var vol int
    for i < j {
        if height[i] > height[j] {
            vol = (j - i) * height[j]
            j--
        } else {
            vol = (j - i) * height[i]
            i++
        }
        if mx < vol {
            mx = vol
        }
    }
    return mx
}
```



## 03 climbing-stairs

 

### python

```python
# Top down - TLE
def climbStairs1(self, n):
    if n == 1:
        return 1
    if n == 2:
        return 2
    return self.climbStairs(n-1)+self.climbStairs(n-2)
 
# Bottom up, O(n) space
def climbStairs2(self, n):
    if n == 1:
        return 1
    res = [0 for i in xrange(n)]
    res[0], res[1] = 1, 2
    for i in xrange(2, n):
        res[i] = res[i-1] + res[i-2]
    return res[-1]

# Bottom up, constant space
def climbStairs3(self, n):
    if n == 1:
        return 1
    a, b = 1, 2
    for i in xrange(2, n):
        tmp = b
        b = a+b
        a = tmp
    return b
    
# Top down + memorization (list)
def climbStairs4(self, n):
    if n == 1:
        return 1
    dic = [-1 for i in xrange(n)]
    dic[0], dic[1] = 1, 2
    return self.helper(n-1, dic)
    
def helper(self, n, dic):
    if dic[n] < 0:
        dic[n] = self.helper(n-1, dic)+self.helper(n-2, dic)
    return dic[n]
    
# Top down + memorization (dictionary)  
def __init__(self):
    self.dic = {1:1, 2:2}
    
def climbStairs(self, n):
    if n not in self.dic:
        self.dic[n] = self.climbStairs(n-1) + self.climbStairs(n-2)
    return self.dic[n]
```



```python
def climbStairs(self, n):
    a, b = 1, 1
    for i in range(n):
        a, b = b, a + b
    return a
```

### go

```python
func climbStairs(n int) int {
    a,b:=1,1
    for i:=0;i<n;i++{
        a,b=b,a+b
    }
    return a
}
```



```go
func climbStairs(n int) int {
    curr := 1
    a := 1
    b := 1
    for i := 2; i <= n; i++ {
        curr = a + b
        a = b
        b = curr
    }
    return curr
}
```

```go
func climbStairs(n int) int {
    res := make([]int, n+1)
    res[0] = 1
    res[1] = 1
    for i := 2; i <= n; i++ {
        res[i] = res[i-1] + res[i-2]
    }
    return res[n]
}
```



## 04 two sum

### python

```python
class Solution(object):
    def twoSum(self, nums, target):        
        dic={}
        for i in enumerate(nums):
            if dic.get(target-nums):
                return [i,dic.get(target-nums)]
            dic[nums]=i
```

```python
class Solution(object):
    def twoSum(self, nums, target):
        if len(nums) <= 1:
            return False
        buff_dict = {}
        for i in range(len(nums)):
            if nums[i] in buff_dict:
                return [buff_dict[nums[i]], i]
            else:
                buff_dict[target - nums[i]] = i
```

### go

```python
func twoSum(nums []int, target int) []int {
    m := make(map[int]int)
    for i, n := range nums {
        _, prs := m[n]
        if prs {
            return []int{m[n], i}
        } else {
            m[target-n] = i
        }
    }
    return nil
}
```



```go
func twoSum(nums []int, target int) []int {
        tmpMap := make(map[int]int)
        for i, num := range nums {
                if _, ok := tmpMap[target-num]; ok {
                        return []int{tmpMap[target-num], i}
                }
                tmpMap[num] = i
        }
        return []int{}

```





## 05 3sum

### python

```python
def threeSum(self, nums):
    res = []
    nums.sort()
    for i in xrange(len(nums)-2):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        l, r = i+1, len(nums)-1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s < 0:
                l +=1 
            elif s > 0:
                r -= 1
            else:
                res.append((nums[i], nums[l], nums[r]))
                while l < r and nums[l] == nums[l+1]:
                    l += 1
                while l < r and nums[r] == nums[r-1]:
                    r -= 1
                l += 1; r -= 1
    return res
```

# linked

## 01 reverse-linked-list

### python

```python
class Solution(object):
    def reverseList(self, head):
        # lterative
        pre=None
        cur=head
        while cur:
            tem=cur.next
            cur.next=pre
            pre=cur
            cur=tem
        return pre
```



```python
class Solution(object):
    def reverseList(self, head):
        return self._serverse(head)
    def _serverse(self,node,pre=None):
        if not node:
            return pre
        n=node.next
        node.next=pre
        return self._serverse(n,node)
```

## 02swap-nodes-in-pairs

### python

```python
class Solution(object):
    def swapPairs(self, head):
        pre,pre.next=self,head
        while pre.next and pre.next.next:
            a=pre.next
            b=a.next
            pre.next,b.next,a.next=b,a,b.next
            pre=a
        return self.next
```



```python
 # Recursively    
def swapPairs(self, head):
    if head and head.next:
        tmp = head.next
        head.next = self.swapPairs(tmp.next)
        tmp.next = head
        return tmp
    return head
```



```python
def swapPairs(self, head):
	if not head or not head.next:
	    return head
	dummy = head
	head = head.next
	dummy.next = head.next
	head.next = dummy
	head.next.next = self.swapPairs(head.next.next)
	return head
```

```python
def swapPairs(self, head):
    if not head or not head.next:
        return head
    first=head.next
    second=head
    second.next=self.swapPairs(first.next)
    first.next=second
	return first
```

## 03 linked-list-cycle

### python

```python
class Solution(object):
    def hasCycle(self, head):
        fast,slow=head,head
        while fast and fast.next:
            fast=fast.next.next
            slow=slow.next
            if fast==slow:return True
        return False
```

## 04 linked-list-cycle-ii

### python

```python
class Solution(object):
    def detectCycle(self, head):
        fast, slow = head, head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow: break
        else:
            return None
        while head !=fast:
            head=head.next
            fast=fast.next
        return head
```

## 05 reverse-nodes-in-k-group

### python

```python
class Solution(object):
    def reverseKGroup(self, head, k):
        dummy = jump = ListNode(0)
        dummy.next = l = r = head
        while True:
            count = 0
            while r and count < k:   # use r to locate the range
                r = r.next
                count += 1
            if count == k:  # if size k satisfied, reverse the inner linked list
                pre, cur = r, l
                for _ in range(k):
                    cur.next, cur, pre = pre, cur.next, cur  # standard reversing
                jump.next, jump, l = pre, l, r  # connect two k-groups
            else:
                return dummy.next
```



```python
class Solution:
    def reverseList(self, head):
        if not head or not head.next:
            return head
        
        prev, cur, nxt = None, head, head
        while cur:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
        return prev 
    
class Solution(object):
    def reverseKGroup(self, head, k):
        count, node = 0, head
        while node and count < k:
            node = node.next
            count += 1
        if count < k: return head
        new_head, prev = self.reverse(head, count)
        head.next = self.reverseKGroup(new_head, k)
        return prev
    
    def reverse(self, head, count):
        prev, cur, nxt = None, head, head
        while count > 0:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
            count -= 1
        return (cur, prev)
```



```python
# Recursively
def reverseKGroup(self, head, k):
    l, node = 0, head
    while node:
        l += 1
        node = node.next
    if k <= 1 or l < k:
        return head
    node, cur = None, head
    for _ in xrange(k):
        nxt = cur.next
        cur.next = node
        node = cur
        cur = nxt
    head.next = self.reverseKGroup(cur, k)
    return node

# Iteratively    
def reverseKGroup(self, head, k):
    if not head or not head.next or k <= 1:
        return head
    cur, l = head, 0
    while cur:
        l += 1
        cur = cur.next
    if k > l:
        return head
    dummy = pre = ListNode(0)
    dummy.next = head
    # totally l//k groups
    for i in xrange(l//k):
        # reverse each group
        node = None
        for j in xrange(k-1):
            nxt = head.next
            head.next = node
            node = head
            head = nxt
        # update nodes and connect nodes
        tmp = head.next
        head.next = node
        pre.next.next = tmp
        tmp1 = pre.next
        pre.next = head
        head = tmp
        pre = tmp1
    return dummy.next
```

# stack

## 01valid-parentheses

### python

```python
 class Solution:
        # @param {string} s
        # @return {boolean}
        def isValid(self, s):
            stack=[]
            for i in s:
                if i in ['(','[','{']:
                    stack.append(i)
                else:
                    if not stack or {')':'(',']':'[','}':'{'}[i]!=stack[-1]:
                        return False
                    stack.pop()
            return not stack
```



```python
def isValid(self, s):
    stack, match = [], {')': '(', ']': '[', '}': '{'}
    for ch in s:
        if ch in match:
            if not (stack and stack.pop() == match[ch]):
                return False
        else:
            stack.append(ch)
    return not stack
```

## 02 min-stack

### python

```python
class MinStack:

def __init__(self):
    self.q = []

# @param x, an integer
# @return an integer
def push(self, x):
    curMin = self.getMin()
    if curMin == None or x < curMin:
        curMin = x
    self.q.append((x, curMin));

# @return nothing
def pop(self):
    self.q.pop()


# @return an integer
def top(self):
    if len(self.q) == 0:
        return None
    else:
        return self.q[len(self.q) - 1][0]


# @return an integer
def getMin(self):
    if len(self.q) == 0:
        return None
    else:
        return self.q[len(self.q) - 1][1]
```

## 03 largest-rectangle-in-histogram

### python

```python
def largestRectangleArea(self, height):
    height.append(0) # very important!!
    stack = [-1]
    ans = 0
    for i in xrange(len(height)):
        while height[i] < height[stack[-1]]:
            h = height[stack.pop()]
            w = i - stack[-1] - 1
            ans = max(ans, h * w)
        stack.append(i)
    height.pop()
    return ans
```



```python
class Solution:
    def largestRectangleArea(self, height):
        n = len(height)
        # left[i], right[i] represent how many bars are >= than the current bar
        left = [1] * n
        right = [1] * n
        max_rect = 0
        # calculate left
        for i in range(0, n):
            j = i - 1
            while j >= 0:
                if height[j] >= height[i]:
                    left[i] += left[j]
                    j -= left[j]
                else: break
        # calculate right
        for i in range(n - 1, -1, -1):
            j = i + 1
            while j < n:
                if height[j] >= height[i]:
                    right[i] += right[j]
                    j += right[j]
                else: break
        for i in range(0, n):
            max_rect = max(max_rect, height[i] * (left[i] + right[i] - 1))
        return max_rect
```

## 04 sliding-window-maximum

### python

```python
class Solution(object):
    def maxSlidingWindow(self, nums, k): #暴力
        n=len(nums)
        if n*k==0:
            return []
        return [max(nums[i:i+k])for i in range(n-k+1)]
```



```python
def maxSlidingWindow(self, nums, k):
    ans = []
    queue = []
    for i, v in enumerate(nums):
        if queue and queue[0] <= i - k:
            queue = queue[1:]
        while queue and nums[queue[-1]] < v:
            queue.pop()
        queue.append(i)
        if i + 1 >= k:
            ans.append(nums[queue[0]])
    return ans
```

```python
def maxSlidingWindow(self, nums, k):
    d = collections.deque()
    out = []
    for i, n in enumerate(nums):
        while d and nums[d[-1]] < n:
            d.pop()
        d += i,
        if d[0] == i - k:
            d.popleft()
        if i >= k - 1:
            out += nums[d[0]],
    return out
```

# 哈希表

## 01 valid-anagram

### python

```python
def isAnagram1(self, s, t):
    dic1, dic2 = {}, {}
    for item in s:
        dic1[item] = dic1.get(item, 0) + 1
    for item in t:
        dic2[item] = dic2.get(item, 0) + 1
    return dic1 == dic2
    
def isAnagram2(self, s, t):
    dic1, dic2 = [0]*26, [0]*26
    for item in s:
        dic1[ord(item)-ord('a')] += 1
    for item in t:
        dic2[ord(item)-ord('a')] += 1
    return dic1 == dic2
    
def isAnagram3(self, s, t):
    return sorted(s) == sorted(t)
```

## 02group-anagrams

### python

```python
def groupAnagrams(self, strs):
    d = {}
    for w in sorted(strs):
        key = tuple(sorted(w))
        d[key] = d.get(key, []) + [w]
    return d.values()
```



# tree

## 01 binary-tree-inorder-traversal

### python

```python
# recursively
def inorderTraversal1(self, root):
    res = []
    self.helper(root, res)
    return res
    
def helper(self, root, res):
    if root:
        self.helper(root.left, res)
        res.append(root.val)
        self.helper(root.right, res)
 
# iteratively       
def inorderTraversal(self, root):
    res, stack = [], []
    while True:
        while root:
            stack.append(root)
            root = root.left
        if not stack:
            return res
        node = stack.pop()
        res.append(node.val)
        root = node.right
```



```python
class Solution:
    def inorderTraversal(self, root):
        result, stack = [], [(root, False)]

        while stack:
            cur, visited = stack.pop()
            if cur:
                if visited:
                    result.append(cur.val)
                else:
                    stack.append((cur.right, False))
                    stack.append((cur, True))
                    stack.append((cur.left, False))

        return result
```



### go



```go
var res []int

func inorderTraversal(root *TreeNode) []int {
	res = make([]int, 0)
	dfs(root)
	return res
}

func dfs(root *TreeNode) {
	if root != nil {
		dfs(root.Left)
		res = append(res, root.Val)
		dfs(root.Right)
	}
}
```



```go
func inorderTraversal(root *TreeNode) []int {
    var res []int
	var stack []*TreeNode
	for root !=nil || len(stack)>0{
		for root !=nil{
			stack=append(stack,root)
			root=root.Left
		}
		pre:=len(stack)-1
		res=append(res,stack[pre].Val)
		root=stack[pre].Right
		stack=stack[:pre]
	}
	return res
}
```



```go
func inorderTraversal(root *TreeNode) []int {
    ret := make([]int, 0)
	if root == nil {
		return ret
	}
	stack := list.New()
	for root != nil || stack.Len() != 0 {
		for root != nil {
			fmt.Println(root.Val)
			stack.PushBack(root)
			root = root.Left
		}
		root = stack.Back().Value.(*TreeNode)
		ret = append(ret, root.Val)
		stack.Remove(stack.Back())
		root = root.Right
	}
	return ret
}
```

## 02 binary-tree-preorder-traversal

### python

```python
class Solution(object):
    def preorderTraversal(self, root):
        if root is None:
            return []
        
        stack, output = [root, ], []
        
        while stack:
            root = stack.pop()
            if root is not None:
                output.append(root.val)
                if root.right is not None:
                    stack.append(root.right)
                if root.left is not None:
                    stack.append(root.left)
        return output

```



```python
# recursively
def preorderTraversal1(self, root):
    res = []
    self.dfs(root, res)
    return res
    
def dfs(self, root, res):
    if root:
        res.append(root.val)
        self.dfs(root.left, res)
        self.dfs(root.right, res)

# iteratively
def preorderTraversal(self, root):
    stack, res = [root], []
    while stack:
        node = stack.pop()
        if node:
            res.append(node.val)
            stack.append(node.right)
            stack.append(node.left)
    return res
```



### go



```go
func preorderTraversal(root *TreeNode) []int {
	if root == nil {
		return nil
	}
	var stack []*TreeNode
	stack = append(stack, root)

	var ret []int
	for len(stack) > 0 {
		p := stack[len(stack)-1]
		stack = stack[0 : len(stack)-1]
		ret = append(ret, p.Val)
		if p.Right != nil {
			stack = append(stack, p.Right)
		}
		if p.Left != nil {
			stack = append(stack, p.Left)
		}
	}

	return ret
}
```



```go
func preorderTraversal(root *TreeNode) []int {
	if root == nil {
		return []int{}
	}

	traversal := make([]int, 0)
	stack := make([]*TreeNode, 0)

	current := root
	for current != nil || len(stack) > 0 {
		for current != nil {
			traversal = append(traversal, current.Val)
			stack = append(stack, current)
			current = current.Left
		}
		current = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		current = current.Right
	}
	return traversal
}
```



## 03 n-ary-tree-postorder-traversal

### python



```python
class Solution(object):
    def postorder(self, root):
        if root is None:
            return []
        
        stack, output = [root, ], []
        while stack:
            root = stack.pop()
            if root is not None:
                output.append(root.val)
            for c in root.children:
                stack.append(c)
        return output[::-1]
```



```python
class Solution(object):
    def postorder(self, root):
        return [] if not root else [j for i in root.children for j in self.postorder(i)] + [root.val]
```



```python
def postorder(self, root):
        """
        :type root: Node
        :rtype: List[int]
        """
        res = []
        if root == None: return res

        def recursion(root, res):
            for child in root.children:
                recursion(child, res)
            res.append(root.val)

        recursion(root, res)
        return res
```



```python
def postorder(self, root):
        res = []
        if root == None: return res

        stack = [root]
        while stack:
            curr = stack.pop()
            res.append(curr.val)
            stack.extend(curr.children)

        return res[::-1]
```

# recursion

## 递归模板

```python
def recursion(level, param1, param2, ...): 
    # recursion terminator 
    if level > MAX_LEVEL: 
	   process_result 
	   return 

    # process logic in current level 
    process(level, data...) 

    # drill down 
    self.recursion(level + 1, p1, ...) 

    # reverse the current level status if needed
```





## 01 generate-parentheses

### python



```python
class Solution(object):
    def generateParenthesis(self, n):
        if not n:
            return []
        left,right,ans=n,n,[]
        self.recursion(left,right,ans,"")
        return ans
    def recursion(self,left,right,ans,s):
        if right<left:
            return 
        #recursion terminato
        if not left and not right:
            ans.append(s)
            return
        # process logic in current level
        # drill down
        if left:
            self.recursion(left-1,right,ans,s+"(")
        if right:
            self.recursion(left , right-1,ans, s + ")")
        # reverse the current level status if needed
```



```python
def generateParenthesis(self, n):
    res = []
    self.dfs(n, n, "", res)
    return res
        
def dfs(self, leftRemain, rightRemain, path, res):
    if leftRemain > rightRemain or leftRemain < 0 or rightRemain < 0:
        return  # backtracking
    if leftRemain == 0 and rightRemain == 0:
        res.append(path)
        return 
    self.dfs(leftRemain-1, rightRemain, path+"(", res)
    self.dfs(leftRemain, rightRemain-1, path+")", res
```



### go

```go
func generateParenthesis(n int) []string {
	m:=map[string]bool{}
	res:=make([]string,0)
	dfs(n,n,"",m)
	for k :=range m{
		res = append(res, k)
	}
	return res
}
func dfs(left,right int,path string,cache map[string]bool)  {
	if left> right || left<0 || right<0{
		return
	}
	if left==0 && right==0{
		cache[path]=true
		return
	}
	dfs(left-1,right,path+"(",cache)
	dfs(left,right-1,path+")",cache)
}
```



## 02 invert-binary-tree

### python

```python
class Solution(object):
    def invertTree(self, root):
        # one silve
        if not root:
            return None
        root.left,root.right=root.right,root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root
        
        #two selve
        if not root:
            return root
        res=[root]
        while res:
            current=res.pop()
            if current:
                current.left, current.right = current.right, current.left
                res+=current.left,current.right
        return root
```



```python
def invertTree(self, root):
    if root:
        invert = self.invertTree
        root.left, root.right = invert(root.right), invert(root.left)
        return root

def invertTree(self, root):
    stack = [root]
    while stack:
        node = stack.pop()
        if node:
            node.left, node.right = node.right, node.left
            stack += node.left, node.right
    return root
```



```python
# recursively
def invertTree1(self, root):
    if root:
        root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root
        
# BFS
def invertTree2(self, root):
    queue = collections.deque([(root)])
    while queue:
        node = queue.popleft()
        if node:
            node.left, node.right = node.right, node.left
            queue.append(node.left)
            queue.append(node.right)
    return root
    
# DFS
def invertTree(self, root):
    stack = [root]
    while stack:
        node = stack.pop()
        if node:
            node.left, node.right = node.right, node.left
            stack.extend([node.right, node.left])
    return root
```

### go

```go
func invertTree(root *TreeNode) *TreeNode {
	if root ==nil{
	return nil
	}
	root.Left,root.Right=root.Right,root.Left
	invertTree(root.Left)
	invertTree(root.Right)
	return root
}
```



```go
func invertTree(root *TreeNode) *TreeNode {
	if root ==nil{
	return nil
	}
    res:= []*TreeNode{}
	res = append(res, root)
	for len(res) >0{
		pre:=len(res)-1
        current:=res[pre]
		res=res[:pre]
		current.Left, current.Right = current.Right, current.Left
		if current.Left !=nil{
			res = append(res, current.Left)
		}
		if current.Right !=nil{
			res = append(res, current.Right)
		}
	}
	return root
}
```

## 03 validate-binary-search-tree

### python

```python
class Solution(object):
    def isValidBST(self, root, lessThan = float('inf'), largerThan = float('-inf')):
        if not root:
            return True
        if root.val <= largerThan or root.val >= lessThan:
            return False
        return self.isValidBST(root.left, min(lessThan, root.val), largerThan) and \
               self.isValidBST(root.right, lessThan, max(root.val, largerThan))
```





```python
# iteratively, in-order traversal
# O(n) time and O(n)+O(lgn) space
def isValidBST(self, root):
    stack, res = [], []
    while True:
        while root:
            stack.append(root)
            root = root.left
        # if root is None or all the nodes have 
        # been traversed and have no confliction 
        if not stack:
            return True
        node = stack.pop()
        # res stores the current values in in-order 
        # traversal order, node.val should larger than
        # the last element in res
        if res and node.val <= res[-1]:
            return False
        res.append(node.val)
        root = node.right
```

### go

```go
func isValidBST(root *TreeNode) bool {
    return helper(root, nil, nil)
}

func helper(root, l, r *TreeNode) bool {
    if root == nil { return true }
    left, right := true, true
    if l != nil { left = l.Val < root.Val }
    if r != nil { right = root.Val < r.Val }
    return left && right && helper(root.Left, l, root) && helper(root.Right, root, r)
}
```



```go
func validBST(t *TreeNode, prev *int) bool { // prev is the min of the current array
    if t == nil {
        return true
    }    
    if bL := validBST(t.Left, prev); !bL || t.Val <= *prev {
        return false
    }
    *prev = t.Val
    return validBST(t.Right, prev)
}
func isValidBST(root *TreeNode) bool {
    prev := math.MinInt64
    return validBST(root, &prev)
}
```



```go
func isValidBST(root *TreeNode) bool {
    stack := []*TreeNode{}
    prev := math.MinInt64
    for len(stack) > 0 || root != nil {
        if root != nil {
            stack = append(stack, root)
            root = root.Left
        } else {
            pop := stack[len(stack)-1]
            stack = stack[:len(stack)-1]
            if pop.Val <= prev {
                return false
            }
            prev = pop.Val
            root = pop.Right
        }
    }
    return true
}
```

## 04 maximum-depth-of-binary-tree

### python

```python
class Solution(object):
    def maxDepth(self, root):
        #递归
        if not root:
            return 0
        else:
            left_hight=self.maxDepth(root.left)
            right_hight=self.maxDepth(root.right)
            return max(left_hight,right_hight)+1
        
        #迭代
        stack=[]
        if root is not None:
            stack.append((1,root))
        depth=0
        while stack:
            current_depth,root=stack.pop()
            if root is not None:
                depth=max(depth,current_depth)
                stack.append((current_depth+1,root.left))
                stack.append((current_depth+1,root.right))
        return depth
```



```python
def maxDepth(self, root):
    return 1 + max(map(self.maxDepth, (root.left, root.right))) if root else 0
```



```python
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        depth = 0
        level = [root] if root else []
        while level:
            depth += 1
            queue = []
            for el in level:
                if el.left:
                    queue.append(el.left)
                if el.right:
                    queue.append(el.right)
            level = queue
            
        return depth
```



```python
# BFS + deque   
def maxDepth(self, root):
    if not root:
        return 0
    from collections import deque
    queue = deque([(root, 1)])
    while queue:
        curr, val = queue.popleft()
        if not curr.left and not curr.right and not queue:
            return val
        if curr.left:
            queue.append((curr.left, val+1))
        if curr.right:
            queue.append((curr.right, val+1))
```



### go



```go
func maxDepth(root *TreeNode) int {
    if root != nil {
        leftDepth := maxDepth(root.Left)
        rightDepth := maxDepth(root.Right)
        if leftDepth > rightDepth {
            return 1+leftDepth
        }
        return 1+rightDepth
    }
    return 0
}
```

## 05 minimum-depth-of-binary-tree

### python

```python
def minDepth(self, root):
    if not root: return 0
    d = map(self.minDepth, (root.left, root.right))
    return 1 + (min(d) or max(d))

def minDepth(self, root):
    if not root: return 0
    d, D = sorted(map(self.minDepth, (root.left, root.right)))
    return 1 + (d or D)
```



```python
# DFS
def minDepth1(self, root):
    if not root:
        return 0
    if None in [root.left, root.right]:
        return max(self.minDepth(root.left), self.minDepth(root.right)) + 1
    else:
        return min(self.minDepth(root.left), self.minDepth(root.right)) + 1
 
# BFS   
def minDepth(self, root):
    if not root:
        return 0
    queue = collections.deque([(root, 1)])
    while queue:
        node, level = queue.popleft()
        if node:
            if not node.left and not node.right:
                return level
            else:
                queue.append((node.left, level+1))
                queue.append((node.right, level+1))
```

### go

```go
func minDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	left, right := minDepth(root.Left), minDepth(root.Right)
	if left == 0 || right == 0 {
		return 1 + left + right
	}
	return 1 + int(math.Min(float64(left), float64(right)))
}
```



```go'
func minDepth(root *TreeNode) int {
    if root ==nil{return 0}
    depth:=1
    node:=[]*TreeNode{root}
    for{
        newNode:=[]*TreeNode{}
        for _,n:=range node{
            if n.Left==nil && n.Right==nil{return depth}
            if n.Left !=nil{newNode=append(newNode,n.Left)}
            if n.Right !=nil{newNode=append(newNode,n.Right)}
        }
        depth++
        node=newNode
    }
    return depth
}
```

# 分治 回溯

## 01 powx-n

### python

```python
class Solution(object):
    def myPow(self, x, n):
        def func(x,n):
            if n==0:
                return 1
            if x == 0:
                return 0
            tem=func(x,n>>1)
            if n & 1:
                return tem*tem*x
            else:
                return tem*tem
        if n >=0:
            res=func(x,n)
        else:
            res=1/func(x,-n)
        return res
```





```python
class Solution:
    def myPow(self, x, n):
        if not n:
            return 1
        if n < 0:
            return 1 / self.myPow(x, -n)
        if n % 2:
            return x * self.myPow(x, n-1)
        return self.myPow(x*x, n/2)

class Solution:
    def myPow(self, x, n):
        if n < 0:
            x = 1 / x
            n = -n
        pow = 1
        while n:
            if n & 1:
                pow *= x
            x *= x
            n >>= 1
        return pow
```



```python
class Solution:
    def myPow(self, a, b):
        if b == 0: return 1
        if b < 0: return 1.0 / self.myPow(a, -b)
        half = self.myPow(a, b // 2)
        if b % 2 == 0:
            return half * half
        else:
            return half * half * a
```



### go

```go
func myPow(x float64, n int) float64 {
    if n ==0{return 1}
    if n <0{return 1/myPow(x,-n)}
    half:=myPow(x,n>>1)
    if n % 2 ==0{
        return half*half
    }
    return half*half*x
}
```



## 02 subsets

### python

```python
class Solution(object):
    def subsets(self, nums):
        res=[]
        n=len(nums)
        def help(i,temp):
            res.append(temp)
            for j in range(i,n):
                help(j+1,temp+[nums[j]])
        help(0,[])
        return res
```



```python
class Solution(object):
    def subsets(self, nums):
        # output=[[]]
        # for num in sorted(nums):
        #     output+=[curr +[num] for curr in output]
        # return output
```



### go

```go
func subsets1(nums []int) [][]int {
	result := make([][]int, 0)
	item := make([]int, 0)

	result = append(result, item)
	generate1(0, nums, &item, &result)
	return result
}

func generate1(i int, nums []int, item *[]int, result *[][]int) {
	if i >= len(nums) {
		return
	}
	*item = append(*item, nums[i])
	temp := make([]int, len(*item))
	for i, v := range *item {
		temp[i] = v
	}
	*result = append(*result, temp)
	generate1(i+1, nums, item, result)
	*item = (*item)[:len(*item)-1]
	generate1(i+1, nums, item, result)
	return
}
// 算法2：回溯，一层递归+一套循环，此处使用闭包函数
func subsets2(nums []int) [][]int {
	result := make([][]int, 0)

	var generate2 func(pos int, num int, item []int)
	generate2 = func(pos int, num int, item []int) {
		if len(item) == num {
			tmp := make([]int, len(item))
			copy(tmp, item)
			result = append(result, tmp)
			return
		}
		for i := pos; i < len(nums); i++ { // 注意：小于nums长度
			item = append(item, nums[i])
			generate2(i+1, num, item)
			item = item[:len(item)-1]
		}
	}

	for i := 0; i <= len(nums); i++ {
		item := make([]int, 0, i) // 注意cap
		generate2(0, i, item)
	}
	return result
}
```

## 03 majority-element

### python

```python
# two pass + dictionary
def majorityElement1(self, nums):
    dic = {}
    for num in nums:
        dic[num] = dic.get(num, 0) + 1
    for num in nums:
        if dic[num] > len(nums)//2:
            return num
    
# one pass + dictionary
def majorityElement2(self, nums):
    dic = {}
    for num in nums:
        if num not in dic:
            dic[num] = 1
        if dic[num] > len(nums)//2:
            return num
        else:
            dic[num] += 1 

# TLE
def majorityElement3(self, nums):
    for i in xrange(len(nums)):
        count = 0
        for j in xrange(len(nums)):
            if nums[j] == nums[i]:
                count += 1
        if count > len(nums)//2:
            return nums[i]
            
# Sotring            
def majorityElement4(self, nums):
    nums.sort()
    return nums[len(nums)//2]
    
# Bit manipulation    
def majorityElement5(self, nums):
    bit = [0]*32
    for num in nums:
        for j in xrange(32):
            bit[j] += num >> j & 1
    res = 0
    for i, val in enumerate(bit):
        if val > len(nums)//2:
            # if the 31th bit if 1, 
            # it means it's a negative number 
            if i == 31:
                res = -((1<<31)-res)
            else:
                res |= 1 << i
    return res
            
# Divide and Conquer
def majorityElement6(self, nums):
    if not nums:
        return None
    if len(nums) == 1:
        return nums[0]
    a = self.majorityElement(nums[:len(nums)//2])
    b = self.majorityElement(nums[len(nums)//2:])
    if a == b:
        return a
    return [b, a][nums.count(a) > len(nums)//2]
    
# the idea here is if a pair of elements from the
# list is not the same, then delete both, the last 
# remaining element is the majority number
def majorityElement(self, nums):
    count, cand = 0, 0
    for num in nums:
        if num == cand:
            count += 1
        elif count == 0:
            cand, count = num, 1
        else:
            count -= 1
    return cand
```

## 04 letter-combinations-of-a-phone-number

### python

```python
class Solution(object):
    def letterCombinations(self, digits):
        if not digits:
            return []
        mapping = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', 
                   '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
        results=['']
        for digit in digits:
            results=[result+d for result in results for d in mapping[digit]]
        return results
```



```python
class Solution:
    # @param {string} digits
    # @return {string[]}
    def letterCombinations(self, digits):
        mapping = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', 
                   '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
        if len(digits) == 0:
            return []
        if len(digits) == 1:
            return list(mapping[digits[0]])
        prev = self.letterCombinations(digits[:-1])
        additional = mapping[digits[-1]]
        return [s + c for s in prev for c in additional] 
```

```python
class Solution(object):
    def letterCombinations(self, digits):
        if not digits:
            return []
        mapping = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', 
                   '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
        res=[]
        self.search("",digits,0,res,mapping)
        return res
    def search(self,s,digits,i,res,mapping):
        # recursion terminator 
        if i==len(digits):
            res.append(s)
            return 
        # process logic in current level 
        letter=mapping.get(digits[i])                  
        #show 
        for j in range(len(letter)):
            self.search(s+letter[j],digits,i+1,res,mapping)
        # reverse the current level status if needed
```

## 05 n-queens

### python

```python
def solveNQueens(self, n):
    def DFS(queens, xy_dif, xy_sum):
        p = len(queens)
        if p==n:
            result.append(queens)
            return None
        for q in range(n):
            if q not in queens and p-q not in xy_dif and p+q not in xy_sum: 
                DFS(queens+[q], xy_dif+[p-q], xy_sum+[p+q])  
    result = []
    DFS([],[],[])
    return [ ["."*i + "Q" + "."*(n-i-1) for i in sol] for sol in result]
```



```python
def solveNQueens(self, n):
    res = []
    self.dfs([-1]*n, 0, [], res)
    return res
 
# nums is a one-dimension array, like [1, 3, 0, 2] means
# first queen is placed in column 1, second queen is placed
# in column 3, etc.
def dfs(self, nums, index, path, res):
    if index == len(nums):
        res.append(path)
        return  # backtracking
    for i in xrange(len(nums)):
        nums[index] = i
        if self.valid(nums, index):  # pruning
            tmp = "."*len(nums)
            self.dfs(nums, index+1, path+[tmp[:i]+"Q"+tmp[i+1:]], res)

# check whether nth queen can be placed in that column
def valid(self, nums, n):
    for i in xrange(n):
        if abs(nums[i]-nums[n]) == n -i or nums[i] == nums[n]:
            return False
    return True
```



```python
class Solution:
# @return a list of lists of string
def solveNQueens(self, n):
    stack, res = [[(0, i)] for i in range(n)], []
    while stack:
        board = stack.pop()
        row = len(board)
        if row == n:
            res.append([''.join('Q' if i == c else '.' for i in range(n))
                        for r, c in board])
        for col in range(n):
            if all(col != c and abs(row-r) != abs(col-c)for r, c in board):
                stack.append(board+[(row, col)])
    return res
```

# 深度优先遍历和广度优先遍历

## DFS 代码模板

**递归写法**

```python
visited = set() 

def dfs(node, visited):
    if node in visited: # terminator
    	# already visited 
    	return 

	visited.add(node) 

	# process current node here. 
	...
	for next_node in node.children(): 
		if next_node not in visited: 
			dfs(next_node, visited)
```

**非递归写法**

```python
def DFS(self, tree): 

	if tree.root is None: 
		return [] 

	visited, stack = [], [tree.root]

	while stack: 
		node = stack.pop() 
		visited.add(node)

		process (node) 
		nodes = generate_related_nodes(node) 
		stack.push(nodes) 

	# other processing work 
	...
```

## BFS 代码模板

```python
def BFS(graph, start, end):
    visited = set()
	queue = [] 
	queue.append([start]) 

	while queue: 
		node = queue.pop() 
		visited.add(node)

		process(node) 
		nodes = generate_related_nodes(node) 
		queue.push(nodes)
	# other processing work   
```

## 01 binary-tree-level-order-traversal

### python

```python
class Solution(object):
    def levelOrder(self, root):
        if not root:
            return []
        ans,level=[],[root]
        while level:
            ans.append([node.val for node in level])
            tem=[]
            for node in level:
                tem.extend([node.left,node.right])
            level=[leaf for leaf in tem if leaf]
        return ans
```

```python
from collections import deque
class Solution:
    def levelOrder(self, root):
        if not root: return []
        queue, res = deque([root]), []
        
        while queue:
            cur_level, size = [], len(queue)
            for i in range(size):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                cur_level.append(node.val)
            res.append(cur_level)
        return res
```

## 02 minimum-genetic-mutation

### python

```python
import collections

def viableMutation(current_mutation, next_mutation):
    changes = 0
    for i in xrange(len(current_mutation)):
        if current_mutation[i] != next_mutation[i]:
            changes += 1
    return changes == 1

class Solution(object):
    def minMutation(self, start, end, bank):
        queue = collections.deque()
        queue.append([start, start, 0]) # current, previous, num_steps
        while queue:
            current, previous, num_steps = queue.popleft()
            if current == end:  # in BFS, the first instance of current == end will yield the minimum
                return num_steps
            for string in bank:
                if viableMutation(current, string) and string != previous:
                    queue.append([string, current, num_steps+1])
        return -1
```



```python
class Solution(object):
    def minMutation(self, start, end, bank):
        bank, v, q = set(bank), {start}, [(start, 0)]
        for g,k in q:
            for s in (g[:i] + cc + g[i+1:] for i,c in enumerate(g) for cc in 'ACGT'):
                if s in bank and s not in v:
                    if s==end:
                        return k+1
                    q.append((s, k+1))
                    v.add(s)
        return -1
```



```python
import collections
class Solution:
    def minMutation(self, start: str, end: str, bank: List[str]) -> int:
        queue = collections.deque()
        queue.append((start,0))
        bankSet = set(bank)
        while queue:
            current,level = queue.popleft()
                
            if current == end:
                return level
                
            for i in range(len(current)):
                for c in "AGCT":
                    mutation = current[:i] + c + current[i+1:]
                    if mutation in bankSet:
                        bankSet.remove(mutation)
                        queue.append((mutation, level+1))  
        return -1
```

## 03 find-largest-value-in-each-tree-row

### python

```python
def findValueMostElement(self, root):
    maxes = []
    row = [root]
    while any(row):
        maxes.append(max(node.val for node in row))
        row = [kid for node in row for kid in (node.left, node.right) if kid]
    return maxes
```



```python
def largestValues(self, root):
    if not root:
        return []
    left = self.largestValues(root.left)
    right = self.largestValues(root.right)
    return [root.val] + map(max, left, right
```

```python
class Solution(object):
    def largestValues(self, root):
        ans = []
        if root is None:
            return ans
        queue  = [root]
        while queue:
            ans.append(max(x.val for x in queue))
            new_queue = []
            for node in queue:
                if node.left:
                    new_queue.append(node.left)
                if node.right:
                    new_queue.append(node.right)
            queue = new_queue
        return ans

class Solution(object):
    def largestValues(self, root):
        self.ans = []
        self.helper(root, 0)
        return self.ans
    
    def helper(self, node, level):
        if not node:
            return
        if len(self.ans) == level:
            self.ans.append(node.val)
        else:
            self.ans[level] = max(self.ans[level], node.val)
        self.helper(node.left, level+1)
        self.helper(node.right, level+1)
```

## 04 word-ladder

### python

```python
class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        wordList = set(wordList)
        queue = collections.deque([[beginWord, 1]])
        while queue:
            word, length = queue.popleft()
            if word == endWord:
                return length
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    next_word = word[:i] + c + word[i+1:]
                    if next_word in wordList:
                        wordList.remove(next_word)
                        queue.append([next_word, length + 1])
        return 0
```

```python
from collections import deque
class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        
        def construct_dict(word_list):
            d = {}
            for word in word_list:
                for i in range(len(word)):
                    s = word[:i] + "_" + word[i+1:]
                    d[s] = d.get(s, []) + [word]
            return d
            
        def bfs_words(begin, end, dict_words):
            queue, visited = deque([(begin, 1)]), set()
            while queue:
                word, steps = queue.popleft()
                if word not in visited:
                    visited.add(word)
                    if word == end:
                        return steps
                    for i in range(len(word)):
                        s = word[:i] + "_" + word[i+1:]
                        neigh_words = dict_words.get(s, [])
                        for neigh in neigh_words:
                            if neigh not in visited:
                                queue.append((neigh, steps + 1))
            return 0
        
        d = construct_dict(wordList or set([beginWord, endWord]))
        return bfs_words(beginWord, endWord, d)
```



```python
class Solution(object):
    def ladderLength(self, start, end, arr):
        arr = set(arr) #avoid TLE
        q = collections.deque([(start, 1)])
        visted = set()
        alpha = string.ascii_lowercase  #'abcd...z'
        while q:
            word, length = q.popleft()
            if word == end:
                return length
            
            for i in range(len(word)):
                for ch in alpha:
                    new_word = word[:i] + ch + word[i+1:]
                    if new_word in arr and new_word not in visted:
                        q.append((new_word, length + 1))
                        visted.add(new_word)
        return 0
```



```python
from collections import defaultdict
class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        if endWord not in wordList or not endWord or not beginWord or not wordList:
            return 0
        L = len(beginWord)
        all_combo_dict = defaultdict(list)
        for word in wordList:
            for i in range(L):
                all_combo_dict[word[:i] + "*" + word[i+1:]].append(word)
        queue = [(beginWord, 1)]
        visited = {beginWord: True}
        while queue:
            current_word, level = queue.pop(0)
            for i in range(L):
                intermediate_word = current_word[:i] + "*" + current_word[i+1:]
                for word in all_combo_dict[intermediate_word]:
                    if word == endWord:
                        return level + 1
                    if word not in visited:
                        visited[word] = True
                        queue.append((word, level + 1))
                all_combo_dict[intermediate_word] = []
        return 0
```



```python
from collections import defaultdict
class Solution(object):
    def __init__(self):
        self.length = 0
        self.all_combo_dict = defaultdict(list)
    def visitWordNode(self, queue, visited, others_visited):
        current_word, level = queue.pop(0)
        for i in range(self.length):
            intermediate_word = current_word[:i] + "*" + current_word[i+1:]
            for word in self.all_combo_dict[intermediate_word]:
                if word in others_visited:
                    return level + others_visited[word]
                if word not in visited:
                    visited[word] = level + 1
                    queue.append((word, level + 1))
        return None
    def ladderLength(self, beginWord, endWord, wordList):

        if endWord not in wordList or not endWord or not beginWord or not wordList:
            return 0
        self.length = len(beginWord)
        for word in wordList:
            for i in range(self.length):
                self.all_combo_dict[word[:i] + "*" + word[i+1:]].append(word)
        queue_begin = [(beginWord, 1)] # BFS starting from beginWord
        queue_end = [(endWord, 1)] # BFS starting from endWord
        visited_begin = {beginWord: 1}
        visited_end = {endWord: 1}
        ans = None
        while queue_begin and queue_end:
            ans = self.visitWordNode(queue_begin, visited_begin, visited_end)
            if ans:
                return ans
            ans = self.visitWordNode(queue_end, visited_end, visited_begin)
            if ans:
                return ans
        return 0
```

## 05 word-ladder-ii

### python

```python
class Solution(object):
    def findLadders(self, beginWord, endWord, wordList):

        wordList = set(wordList)
        res = []
        layer = {}
        layer[beginWord] = [[beginWord]]

        while layer:
            newlayer = collections.defaultdict(list)
            for w in layer:
                if w == endWord: 
                    res.extend(k for k in layer[w])
                else:
                    for i in range(len(w)):
                        for c in 'abcdefghijklmnopqrstuvwxyz':
                            neww = w[:i]+c+w[i+1:]
                            if neww in wordList:
                                newlayer[neww]+=[j+[neww] for j in layer[w]]

            wordList -= set(newlayer.keys())
            layer = newlayer

        return res
```



```python
class Solution(object):
    def findLadders(self,beginWord, endWord, wordList):
        tree, words, n = collections.defaultdict(set), set(wordList), len(beginWord)
        if endWord not in wordList: return []
        found, bq, eq, nq, rev = False, {beginWord}, {endWord}, set(), False
        while bq and not found:
            words -= set(bq)
            for x in bq:
                for y in [x[:i]+c+x[i+1:] for i in range(n) for c in 'qwertyuiopasdfghjklzxcvbnm']:
                    if y in words:
                        if y in eq: 
                            found = True
                        else: 
                            nq.add(y)
                        tree[y].add(x) if rev else tree[x].add(y)
            bq, nq = nq, set()
            if len(bq) > len(eq): 
                bq, eq, rev = eq, bq, not rev
        def bt(x): 
            return [[x]] if x == endWord else [[x] + rest for y in tree[x] for rest in bt(y)]
        return bt(beginWord)
```



=========================================================================================

# Tree

## python



````python
class TreeNone:
    def __init__(self,val):
        self.val=val
        self.left,self.right=None,None
````



```python
class Tree():
    def __init__(self):#构造出一颗空的二叉树
        self.root = None #root指向第一个节点的地址，如果root指向了None，则意味着该二叉树为空           
    def forward(self,root):
        if root == None:
            return
        print(root.item)
        self.forward(root.left)
        self.forward(root.right)
        
    def middle(self,root):
        if root == None:
            return
        self.middle(root.left)
        print(root.item)
        self.middle(root.right)
    def back(self,root):
        if root == None:
            return
        self.back(root.left)
        self.back(root.right)
        print(root.item)
```



## go



```go
type Tree struct {
	value int
	left *Tree
	right *Tree
} 
```



## java



```java
public class TreeNode{
    public int val;
    public TreeNode left,right;
    public TreeNode(int val){
        this.val=val;
        this.left=null;
        this.right=null;
    }
}
```



## C++



```c++
struct TreeNode{
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x):val(x),left(NULL),right(NULL){}
};
```

# 分治代码模板

```python
def divide_conquer(problem, param1, param2, ...): 
  # recursion terminator 
  if problem is None: 
	print_result 
	return 

  # prepare data 
  data = prepare_data(problem) 
  subproblems = split_problem(problem, data) 

  # conquer subproblems 
  subresult1 = self.divide_conquer(subproblems[0], p1, ...) 
  subresult2 = self.divide_conquer(subproblems[1], p1, ...) 
  subresult3 = self.divide_conquer(subproblems[2], p1, ...) 
  …

  # process and generate the final result 
  result = process_result(subresult1, subresult2, subresult3, …)
	
  # revert the current level states
```

