import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import time
import random

@dataclass
class ARLCParams:
    """Parameters for ARLC algorithm (LWE-based)"""
    n: int = 256       # Dimension of secret key
    m: int = 512       # Number of samples (rows of A)
    q: int = 32768     # Modulus (2^15) - increased for better error tolerance
    eta: int = 4       # Error bound for key and key-generation errors - slightly increased
    p: int = 256       # Message space (each message symbol in 0...255)
    delta: int = 32768 // 256  # Scaling factor (delta = 128) - increased for better error tolerance
    r_weight: int = 64  # Number of ones in the sparse binary vector r for encryption - increased for better error distribution

class ARLC:
    """Avionics-Resilient Lattice Cryptography implementation using LWE-based encryption"""
    
    def __init__(self, params: Optional[ARLCParams] = None):
        self.params = params or ARLCParams()
        # Set a fixed seed for reproducibility
        np.random.seed(42)
        random.seed(42)
        # Generate public matrix A of shape (m, n)
        self.A = np.random.randint(0, self.params.q, (self.params.m, self.params.n))
        
    def _generate_error(self, size: Tuple[int, ...]) -> np.ndarray:
        """
        Generate an error term sampled uniformly from [-eta, eta].
        """
        return np.random.randint(-self.params.eta, self.params.eta + 1, size)
    
    def generate_keypair(self) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Generate a public/private key pair.
        Secret key s ∈ ℤ₍q₎ⁿ is sampled from a small error distribution.
        Error vector e ∈ ℤ₍q₎ᵐ is sampled similarly.
        Public key is (A, b) with b = A · s + e.
        """
        s = self._generate_error((self.params.n,))      # Secret key (n,)
        e = self._generate_error((self.params.m,))      # Error vector (m,)
        b = (np.dot(self.A, s) + e) % self.params.q     # Public vector b (m,)
        public_key = (self.A, b)
        return public_key, s
    
    def _encode_message_symbol(self, m: int) -> int:
        """
        Scale a message symbol m (0 ≤ m < p) into lattice space.
        We center the message by adding p//2 and then multiply by delta.
        """
        if not 0 <= m < self.params.p:
            raise ValueError(f"Message symbol {m} is not in valid range [0, {self.params.p})")
        return ((m + self.params.p // 2) * self.params.delta) % self.params.q
    
    def _decode_message_symbol(self, m_scaled: int) -> int:
        """
        Inverse of _encode_message_symbol.
        Given m_scaled, recover m by dividing by delta and subtracting p//2.
        """
        # Ensure m_scaled is in the correct range
        m_scaled = m_scaled % self.params.q
        
        # Compute the approximate message value
        m_approx = int(round(m_scaled / self.params.delta)) - (self.params.p // 2)
        
        # Ensure the result is in the valid range
        m_rec = m_approx % self.params.p
        
        # Additional validation to ensure the recovered value is reasonable
        if abs(m_approx - m_rec) > self.params.delta // 4:
            # If the error is too large, we might have wrapped around
            # Try the other possible value
            alt_m = (m_rec + self.params.p) % self.params.p
            if abs(m_approx - alt_m) < abs(m_approx - m_rec):
                m_rec = alt_m
                
        return m_rec
    
    def _generate_sparse_r(self) -> np.ndarray:
        """
        Generate a sparse binary vector r ∈ {0,1}^m with exactly r_weight ones.
        """
        r = np.zeros(self.params.m, dtype=np.int64)
        ones_positions = random.sample(range(self.params.m), self.params.r_weight)
        r[ones_positions] = 1
        return r
    
    def encrypt(self, message: str, public_key: Tuple[np.ndarray, np.ndarray]) -> List[Tuple[np.ndarray, int]]:
        """
        Encrypt a message using the receiver's public key.
        
        For each message symbol, we:
          - Scale the message symbol to lattice space.
          - Generate a sparse binary vector r (of length m) with a fixed Hamming weight.
          - Compute U = Aᵀ · r ∈ ℤ₍q₎ⁿ.
          - Compute V = rᵀ · b + m_scaled ∈ ℤ₍q₎.
        
        Returns:
            A list of ciphertext tuples, one per message symbol.
            Each tuple is (U, V) with U ∈ ℤ₍q₎ⁿ and V ∈ ℤ₍q₎.
        """
        A, b = public_key
        ciphertext = []
        
        for ch in message:
            m_val = ord(ch)
            m_scaled = self._encode_message_symbol(m_val)
            r = self._generate_sparse_r()  # vector in {0,1}^m with weight r_weight
            
            # Compute U and V with additional error checking
            U = np.dot(A.T, r) % self.params.q  # U is in Z_q^n
            V = (np.dot(r, b) + m_scaled) % self.params.q
            
            # Validate the ciphertext components
            if np.any(U >= self.params.q) or V >= self.params.q:
                raise ValueError("Ciphertext components exceed modulus q")
                
            ciphertext.append((U, int(V)))
            
        return ciphertext
    
    def decrypt(self, ciphertext: List[Tuple[np.ndarray, int]], secret_key: np.ndarray) -> str:
        """
        Decrypt the ciphertext.
        
        For each ciphertext tuple (U, V):
          - Compute prod = Uᵀ · s.
          - Subtract prod from V to recover m_scaled (plus a small error).
          - Apply inverse scaling to recover the original message symbol.
        
        Returns:
            The decrypted message string.
        """
        message_chars = []
        
        for U, V in ciphertext:
            # Validate input
            if np.any(U >= self.params.q) or V >= self.params.q:
                raise ValueError("Invalid ciphertext: components exceed modulus q")
            
            # Compute the product and ensure it's in the correct range
            prod = int(np.dot(U, secret_key) % self.params.q)
            
            # Recover the scaled message
            m_scaled = (V - prod) % self.params.q
            
            # Decode the message symbol
            m_rec = self._decode_message_symbol(m_scaled)
            
            # Convert to character and validate
            try:
                ch = chr(m_rec)
                if not ch.isprintable():
                    ch = '?'  # Replace non-printable characters
            except ValueError:
                ch = '?'
                
            message_chars.append(ch)
            
        return ''.join(message_chars)
