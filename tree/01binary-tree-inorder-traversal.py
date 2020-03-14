class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def inorderTraversal(self, root):  # recursively
        res=[]
        self.help(root,res)
        return res
    def help(self,root,res):
        if root:
            self.help(root.left,res)
            res.append(root.val)
            self.help(root.right,root)
    def inorderTraversal1(self, root): # iteratively
        res,stack=[],[]
        while True:
            while root:
                stack.append(root)
                root=root.left
            if not stack:
                return res
            node=stack.pop()
            res.append(node.val)
            root=root.right
