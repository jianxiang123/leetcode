class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def swapPairs(self, head):
        pre,pre.next=self,head
        while pre.next and pre.next.next:
            a=pre.next
            b=a.next
            pre.next,b.next,a.next=b,a,b.next
            pre=a
        return self.next
    def swapPairs1(self, head):
        if not head or not head.next:
            return head
        first=head.next
        second=head
        second.next=self.swapPairs1(first.next)
        first.next=second
        return first

    def swapPairs2(self, head):
        if not head or not head.next:
            return head
        dummy = head
        head = head.next
        dummy.next = head.next
        head.next = dummy
        head.next.next = self.swapPairs2(head.next.next)
        return head