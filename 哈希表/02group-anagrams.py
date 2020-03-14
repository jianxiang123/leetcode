from collections import defaultdict
class Solution(object):
    def groupAnagrams1(self, strs):
        ans=defaultdict(list)
        for ch in strs:
            ans["".join(sorted(ch))].append(ch)
        return ans.values()

    def groupAnagrams(self, strs):
        d = {}
        for w in sorted(strs):
            key =tuple(sorted(w))
            d[key]=d.get(key,[])+[w]
        return d.values()

