test_case_1=[10, 5, 2, 20, 1]
test_case_2=[1, 2, 1]
test_case_3=[1, 5, 10, 5, 1]

def matrix_chain_multiplication(dimensions):
    """
    Identifies optimal way to multiply a chain of matrices.
    
    Args:
        dimensions: List of dimensions where matrices are:
                   A1: dimensions[0] x dimensions[1]
                   A2: dimensions[1] x dimensions[2]
                   ...
                   An: dimensions[n-1] x dimensions[n]
    
    Returns:
        Tuple of (minimum cost, optimal parenthesization string)
    """
    n = len(dimensions)-1
    
    m = [[0] * n for _ in range(n)]
    
    s = [[0] * n for _ in range(n)]
    
    # L is the chain length
    for L in range(2, n + 1):
        for i in range(n - L + 1):
            j = i + L - 1
            m[i][j] = float('inf')

            for k in range(i, j):
                cost = (m[i][k] + m[k + 1][j] + 
                       dimensions[i] * dimensions[k + 1] * dimensions[j + 1])
                
                if cost < m[i][j]:
                    m[i][j] = cost
                    s[i][j] = k
    
    # Construct the optimal parenthesization
    def construct_parenthesization(i, j):
        if i == j:
            return f"A{i + 1}"
        
        k = s[i][j]
        left = construct_parenthesization(i, k)
        right = construct_parenthesization(k + 1, j)
        
        return f"({left} × {right})"
    
    optimal_order = construct_parenthesization(0, n - 1)
    
    return m[0][n - 1], optimal_order

if __name__ == "__main__":
    test_cases = [test_case_1, test_case_2, test_case_3]
    
    for i, case in enumerate(test_cases, 1):
        cost, order = matrix_chain_multiplication(case)
        print(f"Test Case {i}: Dimensions = {case}")
        print(f"Minimum number of multiplications: {cost}")
        print(f"Optimal Parenthesization: {order}\n")