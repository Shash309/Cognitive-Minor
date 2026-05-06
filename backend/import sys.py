import sys
input = sys.stdin.readline

MOD = 10**9 + 7

# Matrix multiplication (3x3)
def matmul(A, B):
    C = [[0]*3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                C[i][j] = (C[i][j] + A[i][k] * B[k][j]) % MOD
    return C

# Matrix exponentiation
def matpow(A, power):
    result = [[1 if i == j else 0 for j in range(3)] for i in range(3)]
    while power:
        if power % 2:
            result = matmul(result, A)
        A = matmul(A, A)
        power //= 2
    return result

# Count ways for '?' positions
def count_ways(n):
    cnt = [4, 3, 3]  # mod 3 residues for digits 0-9
    
    A = [[0]*3 for _ in range(3)]
    for i in range(3):
        for r in range(3):
            A[i][(i + r) % 3] += cnt[r]
    
    A = matpow(A, n)
    return [A[0][i] for i in range(3)]


def solve():
    s = input().strip()
    n = len(s)
    total = 0
    
    for d in range(10):  # value for '*'
        arr = list(s)
        
        # Replace '*'
        for i in range(n):
            if arr[i] == '*':
                arr[i] = str(d)
        
        # Last digit condition
        if arr[-1] == '?':
            last_options = [0, 5]
        else:
            if int(arr[-1]) % 5 != 0:
                continue
            last_options = [int(arr[-1])]
        
        sum_mod = 0
        q_count = 0
        
        for i in range(n):
            if arr[i] == '?':
                q_count += 1
            else:
                sum_mod = (sum_mod + int(arr[i])) % 3
        
        ways = 0
        
        if arr[-1] == '?':
            q_count -= 1
            dp = count_ways(q_count)
            
            for last in last_options:
                needed = (- (sum_mod + last) % 3) % 3
                ways = (ways + dp[needed]) % MOD
        
        else:
            dp = count_ways(q_count)
            needed = (-sum_mod) % 3
            ways = dp[needed]
        
        total = (total + ways) % MOD
    
    print(total)


if __name__ == "__main__":
    solve()