class Solution:
    def isValid(self, s):
        stack = []
        for i in s:
            if i in ['(', '[', '{']:
                stack.append(i)
            else:
                if not stack or {')': '(', ']': '[', '}': '{'}[i] != stack[-1]:
                    return False
                stack.pop()
        return not stack