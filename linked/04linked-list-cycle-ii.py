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
            head=head.ext
            fast=fast.next
        return head