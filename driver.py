import numpy as np
from arlc import ARLC, ARLCParams
import time
from typing import Tuple, List

def print_public_key(public_key: tuple):
    """Pretty print the public key (A, b) with their shapes and sample elements"""
    A, b = public_key
    print("Public Key:")
    print("-" * 50)
    print("Matrix A:")
    print(f"Shape: {A.shape}")
    print(f"First row (first 10 elements): {A[0, :10]}")
    print("-" * 50)
    print("Vector b:")
    print(f"Shape: {b.shape}")
    print(f"First few elements: {b[:10]}")
    print("-" * 50 + "\n")

def print_vector(name: str, vector: np.ndarray):
    """Pretty print a vector with its shape"""
    print(f"{name}:")
    print("-" * 50)
    print(f"Shape: {vector.shape}")
    print(f"First few elements: {vector[:10]}")
    print("-" * 50 + "\n")

def print_ciphertext(ciphertext: List[Tuple[np.ndarray, int]]):
    """Pretty print ciphertext information for the first symbol"""
    U, V = ciphertext[0]
    print("Ciphertext for first symbol:")
    print("-" * 50)
    print("U (first 10 elements):", U[:10])
    print("V:", V)
    print("-" * 50 + "\n")

def test_message(message: str, arlc: ARLC) -> Tuple[bool, float, float, float, str, List[Tuple[np.ndarray, int]]]:
    """Test encryption and decryption of a single message"""
    # Generate key pair
    start_time = time.time()
    public_key, secret_key = arlc.generate_keypair()
    key_gen_time = time.time() - start_time
    
    # Encryption
    start_time = time.time()
    ciphertext = arlc.encrypt(message, public_key)
    encrypt_time = time.time() - start_time
    
    # Decryption
    start_time = time.time()
    decrypted_message = arlc.decrypt(ciphertext, secret_key)
    decrypt_time = time.time() - start_time
    
    # Verify
    success = message == decrypted_message
    
    if not success:
        print("\nDecryption failed!")
        print(f"Original:  {message}")
        print(f"Decrypted: {decrypted_message}")
        # Find differences
        for i, (orig, dec) in enumerate(zip(message, decrypted_message)):
            if orig != dec:
                print(f"Difference at position {i}: '{orig}' -> '{dec}'")
    
    return success, key_gen_time, encrypt_time, decrypt_time, decrypted_message, ciphertext

def main():
    # Initialize ARLC with default parameters
    arlc = ARLC()
    
    print("ARLC Implementation Test")
    print("=" * 50)
    
    # Test messages with different characteristics
    test_messages = [
        "Hello, World!",
        "Aircraft A to B: Requesting permission to change altitude to 35,000ft",
        "Testing special characters: !@#$%^&*()",
        "Testing numbers: 1234567890",
        "Testing spaces:   multiple   spaces   here  ",
        "Testing newlines:\nline1\nline2",
    ]
    
    total_tests = len(test_messages)
    successful_tests = 0
    total_key_gen_time = 0
    total_encrypt_time = 0
    total_decrypt_time = 0
    
    for i, message in enumerate(test_messages, 1):
        print(f"\nTest {i}/{total_tests}")
        print("-" * 50)
        print(f"Original Message: {message}")
        
        success, key_gen_time, encrypt_time, decrypt_time, decrypted_message, ciphertext = test_message(message, arlc)
        
        print("\nTiming Breakdown:")
        print(f"  Key Generation: {key_gen_time:.4f} seconds")
        print(f"  Encryption: {encrypt_time:.4f} seconds")
        print(f"  Decryption: {decrypt_time:.4f} seconds")
        print(f"  Total Time: {(key_gen_time + encrypt_time + decrypt_time):.4f} seconds")
        
        print("\nResults:")
        print(f"  Success: {'✓' if success else '✗'}")
        print(f"  Decrypted Message: {decrypted_message}")
        
        if success:
            successful_tests += 1
        
        total_key_gen_time += key_gen_time
        total_encrypt_time += encrypt_time
        total_decrypt_time += decrypt_time
    
    # Print summary
    print("\nTest Summary")
    print("=" * 50)
    print(f"Total Tests: {total_tests}")
    print(f"Successful Tests: {successful_tests}")
    print(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")
    print("\nAverage Performance:")
    print(f"Key Generation: {total_key_gen_time/total_tests:.4f} seconds")
    print(f"Encryption: {total_encrypt_time/total_tests:.4f} seconds")
    print(f"Decryption: {total_decrypt_time/total_tests:.4f} seconds")
    print(f"Total Average Time: {(total_key_gen_time + total_encrypt_time + total_decrypt_time)/total_tests:.4f} seconds")
    
    # Print security parameters
    print("\nSecurity Parameters")
    print("=" * 50)
    print(f"Lattice dimension (n): {arlc.params.n}")
    print(f"Public key rows (m): {arlc.params.m}")
    print(f"Modulus (q): {arlc.params.q}")
    print(f"Error parameter (eta): {arlc.params.eta}")
    print(f"Message space (p): {arlc.params.p}")
    print(f"Scaling factor (delta): {arlc.params.delta}")
    print(f"Sparse vector weight: {arlc.params.r_weight}")
    print(f"Estimated security level: ~128 bits")

if __name__ == "__main__":
    main()
