class Solution(object):
    def hasCycle(self, head): #快慢指针
        fast, slow = head, head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow: return True
        return False
