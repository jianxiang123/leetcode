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
    # d递归
    def reverseList1(self, head):
        return self._serverse1(head)

    def _serverse1(self, node, pre=None):
        if not node:
            return pre
        n = node.next
        node.next = pre
        return self._serverse1(n, node)