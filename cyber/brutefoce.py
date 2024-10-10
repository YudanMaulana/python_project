import itertools
import string

def brute_force(password):
    characters = string.ascii_letters + string.digits
    for length in range(1, 6):
        for attempt in itertools.product(characters, repeat=length):
            attempt_password = ''.join(attempt)
            print(f"Trying : {attempt_password}")
            if attempt_password == password:
                return f"Password found: {attempt_password}"
    return "Password not found"

# password example
password = "abcde"
hasil = brute_force(password)
print(hasil)
