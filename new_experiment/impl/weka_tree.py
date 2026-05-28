import re
import numpy as np


class Node:
    def __init__(self, feature=None, op=None, threshold=None, prediction=None):
        self.feature = feature
        self.op = op
        self.threshold = threshold
        self.prediction = prediction
        self.left = None
        self.right = None


def _parse_line(line):
    # remove leading bars and spaces
    m = re.match(r"^([| ]*)(.*)$", line)
    prefix, body = m.groups()
    depth = prefix.count('|')
    body = body.strip()
    # leaf line contains ':'
    if ':' in body:
        cond, rest = body.split(':', 1)
        cond = cond.strip()
        # if cond is empty, it's a leaf
        if cond == '':
            # rest begins with class label
            pred = rest.strip().split()[0]
            return depth, None, None, None, pred
        else:
            # condition may be like 'col2 <= 0.348101' or 'col2 > 0.348101: 1 (420.0/31.0)'
            if '<=' in cond:
                feat, thr = cond.split('<=')
                return depth, feat.strip(), '<=', float(thr.strip()), None
            elif '>' in cond:
                feat, thr = cond.split('>')
                return depth, feat.strip(), '>', float(thr.strip()), None
            else:
                return depth, cond, None, None, None
    else:
        # internal node without immediate leaf, like 'col2 > 0.348101'
        if '<=' in body:
            feat, thr = body.split('<=')
            return depth, feat.strip(), '<=', float(thr.strip()), None
        elif '>' in body:
            feat, thr = body.split('>')
            return depth, feat.strip(), '>', float(thr.strip()), None
        else:
            return depth, None, None, None, None


def parse_j48_tree(text):
    """Parse the textual J48 tree from Weka stdout and return a root Node.

    The parser handles the simple printed tree format with '|' indentation.
    """
    lines = text.splitlines()
    # find the first line that looks like a split (contains '<=' or '>')
    tree_lines = []
    started = False
    for ln in lines:
        if '<=' in ln or '>' in ln:
            started = True
        if started:
            # stop at blank line before 'Number of Leaves' or 'Time taken'
            if ln.strip() == '':
                break
            tree_lines.append(ln.rstrip())
    if not tree_lines:
        return None

    # build nodes stack by depth
    stack = []
    root = None
    for ln in tree_lines:
        depth, feat, op, thr, pred = _parse_line(ln)
        node = Node(feature=feat, op=op, threshold=thr, prediction=pred)
        if depth == 0:
            root = node
            stack = [(depth, node)]
        else:
            # find parent at depth-1
            while stack and stack[-1][0] >= depth:
                stack.pop()
            if not stack:
                stack = [(depth - 1, root)]
            parent = stack[-1][1]
            # attach left if empty else right
            if parent.left is None:
                parent.left = node
            else:
                parent.right = node
            stack.append((depth, node))
    return root


def _eval_node(node, x, feat_map):
    if node is None:
        return None
    if node.prediction is not None and node.feature is None:
        return node.prediction
    # get feature value
    fname = node.feature
    if fname not in feat_map:
        # try to parse numeric index 'colN'
        if fname.startswith('col'):
            idx = int(fname[3:]) - 1
        else:
            raise KeyError(f"Unknown feature {fname}")
    else:
        idx = feat_map[fname]
    val = x[idx]
    if node.op == '<=':
        if val <= node.threshold:
            if node.left is None:
                return node.prediction
            return _eval_node(node.left, x, feat_map)
        else:
            if node.right is None:
                return node.prediction
            return _eval_node(node.right, x, feat_map)
    elif node.op == '>':
        if val > node.threshold:
            if node.left is None:
                return node.prediction
            return _eval_node(node.left, x, feat_map)
        else:
            if node.right is None:
                return node.prediction
            return _eval_node(node.right, x, feat_map)
    else:
        # unknown op -> return prediction
        return node.prediction


def predict_leaves(root, X):
    """Return a leaf id (int) for each row in X using parsed J48 tree root."""
    X = np.asarray(X)
    # build feature name map: col1 -> 0, col2 -> 1, etc.
    feat_map = {}
    # scan tree for feature names
    def collect(n):
        if n is None:
            return
        if n.feature is not None and n.feature.startswith('col'):
            try:
                idx = int(n.feature[3:]) - 1
                feat_map[n.feature] = idx
            except Exception:
                pass
        collect(n.left)
        collect(n.right)

    collect(root)

    ids = []
    uniq = {}
    next_id = 0
    for i in range(X.shape[0]):
        pred = _eval_node(root, X[i], feat_map)
        # use object id of terminal node represented by prediction+path
        key = str(pred)
        if key not in uniq:
            uniq[key] = next_id
            next_id += 1
        ids.append(uniq[key])
    return np.array(ids, dtype=int)
