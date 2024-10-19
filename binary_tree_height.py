class TreeNode:
    def __init__(self, value=0, left=None, right=None) -> None:
        self.value = value
        self.left = left
        self.right = right

def tree_height(root):
    if root is None:
        return 0
    else:
        left_height = tree_height(root.left)
        right_height = tree_height(root.right)
        return max(left_height, right_height) + 1
    
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left = TreeNode(4)
root.right = TreeNode(5)

print(f"The height of the binary tree is {tree_height(root)}")