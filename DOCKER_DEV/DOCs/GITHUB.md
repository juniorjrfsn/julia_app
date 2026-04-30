<https://github.com/juniorjrfsn/chatbot_ageprev_py.git>
<git@github.com>:juniorjrfsn/chatbot_ageprev_py.git
gh repo clone juniorjrfsn/chatbot_ageprev_py

# gpg --full-generate-key

gpg --default-new-key-algo rsa4096 --gen-key
gpg --list-secret-keys --keyid-format=long
$ gpg --list-secret-keys --keyid-format=long
/Users/hubot/.gnupg/secring.gpg
------------------------------------

sec   4096R/3AA5C34371567BD2 2016-03-10 [expires: 2017-03-10]
uid                          Hubot <hubot@example.com>
ssb   4096R/4BB6D45482678BE3 2016-03-10

gpg --list-secret-keys --keyid-format=long
$ gpg --armor --export 3AA5C34371567BD2

# Prints the GPG public key, in ASCII armor format

<git@github.com>:juniorjrfsn/chatbot_ageprev_py.git
<https://github.com/juniorjrfsn/chatbot_ageprev_py.git>
gh repo clone juniorjrfsn/chatbot_ageprev_py

njunior@ALAG074836 MINGW64 /d/AGEPREV-PROJETOS/DOCKER_DEV
$ gpg --full-generate-key
gpg (GnuPG) 2.4.9; Copyright (C) 2025 g10 Code GmbH
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.

Please select what kind of key you want:
   (1) RSA and RSA
   (2) DSA and Elgamal
   (3) DSA (sign only)
   (4) RSA (sign only)
   (9) ECC (sign and encrypt) *default*
  (10) ECC (sign only)
  (14) Existing key from card
Your selection? 1
RSA keys may be between 1024 and 4096 bits long.
What keysize do you want? (3072)
Requested keysize is 3072 bits
Please specify how long the key should be valid.
         0 = key does not expire
      <n>  = key expires in n days
      <n>w = key expires in n weeks
      <n>m = key expires in n months
      <n>y = key expires in n years
Key is valid for? (0) 0
Key does not expire at all
Is this correct? (y/N) Y

GnuPG needs to construct a user ID to identify your key.

Real name: JuniorNSF
Email address: <junior.jrfsn@gmail.com>
Comment:
You selected this USER-ID:
    "JuniorNSF <junior.jrfsn@gmail.com>"

Change (N)ame, (C)omment, (E)mail or (O)kay/(Q)uit? O
We need to generate a lot of random bytes. It is a good idea to perform
some other action (type on the keyboard, move the mouse, utilize the
disks) during the prime generation; this gives the random number
generator a better chance to gain enough entropy.
We need to generate a lot of random bytes. It is a good idea to perform
some other action (type on the keyboard, move the mouse, utilize the
disks) during the prime generation; this gives the random number
generator a better chance to gain enough entropy.
gpg: directory '/c/Users/njunior/.gnupg/openpgp-revocs.d' created
gpg: revocation certificate stored as '/c/Users/njunior/.gnupg/openpgp-revocs.d/2E5C0E9F32F4A5C450050F17BF807559C9AA801C.rev'
public and secret key created and signed.

pub   rsa3072 2026-04-30 [SC]
      2E5C0E9F32F4A5C450050F17BF807559C9AA801C
uid                      JuniorNSF <junior.jrfsn@gmail.com>
sub   rsa3072 2026-04-30 [E]

GPG
<junior.jrfsn@gmail.com>
@jrfsn@g202604

$ gpg --list-secret-keys --keyid-format=long
gpg: checking the trustdb
gpg: marginals needed: 3  completes needed: 1  trust model: pgp
gpg: depth: 0  valid:   1  signed:   0  trust: 0-, 0q, 0n, 0m, 0f, 1u
[keyboxd]
---------

sec   rsa3072/BF807559C9AA801C 2026-04-30 [SC]
      2E5C0E9F32F4A5C450050F17BF807559C9AA801C
uid                 [ultimate] JuniorNSF <junior.jrfsn@gmail.com>
ssb   rsa3072/8F0E333C70DC804B 2026-04-30 [E]

:-----BEGIN PGP PUBLIC KEY BLOCK-----
Comment: This is a revocation certificate

iQG2BCABCgAgFiEELlwOnzL0pcRQBQ8Xv4B1WcmqgBwFAmnzlDgCHQAACgkQv4B1
WcmqgBzN0Av8CkfsQ65Go+52Z2Y+eQGfxOVumk4mQqQsXQwJPKFnGjex8oJZbNz0
v0686JVi1yxOdMREUmPdaa2RvW9a7j+wbonpmMEroVtNR9FJhka9OD9FCK10ZuLt
JRLG3iENamOtMefugL+qv4TlxJy78Y9BLniNnnmsjl6ScNzHfoob/MOppWcbGc94
0T+nSkD4MEG8ZbrUv0DREF36cEvG6oeEU5Yr+lj61Yk1ZUnZCZ5oLwZOQnEs0nh3
U8VITVlnVGvYI/3VWIjvLYal+6BgsKHfSugfAbPeSnqbp2JRw2TYv+iDO9bNFi4t
w85itC4MtDjx6qGcjE4SqIulU+A9VFdZhcOHL2vQEl/E26q7lVLisZzercnGof2U
JuGg16x5Dy092BPC8u0lw+4ymYPGCF+etQ41MMyTefStmMPH6RoMu556mqrgKqbU
e8sKZmxb9xFq572yMP7YiaekXkDEm1WQrasDtKhzDnaDH3ZCDvEXkk7lZ+XyQ2Z8
zPRPN+wErZN6
=7MJk
-----END PGP PUBLIC KEY BLOCK-----

GITHUB TOKEN
chatbot
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
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
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

<http://172.17.0.2:8000/chat/>
<http://127.0.0.1:8000/chat/>
<http://127.0.0.1:8000/treinar/>
