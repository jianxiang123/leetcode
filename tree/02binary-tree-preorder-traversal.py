class Solution(object):
    def preorderTraversal(self, root): #resursively
        res=[]
        self.help(root,res)
        return res
    def help(self,root,res):
        if root:
            res.append(root.val)
            self.help(root.left)
            self.help(root.right)

    def preorderTraversal1(self, root):# iteratively
        stack,res=[root],[]
        while stack:
            node=stack.pop()
            if node:
                res.append(node.val)
                stack.append(node.left)
                stack.append(node.right)
        return res
