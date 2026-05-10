# GAN-Implementations

Includes

## 1. Basic GANs

### Generative Adversarial Network (GAN) Objective Function

The standard GAN minimax objective is:

$$
\min_G \max_D V(D, G)
=
\mathbb{E}_{x \sim p_{\text{data}}(x)} [\log D(x)]
+
\mathbb{E}_{z \sim p_z(z)} [\log(1 - D(G(z)))]
$$

Where:

- \( G \) = Generator
- \( D \) = Discriminator
- $$x \sim p_{\text{data}}(x)$$ means $$x$$ is sampled from the real data distribution
- $$z \sim p_z(z)$$ means $$z$$ is sampled from the latent noise distribution

---

## Discriminator Loss

$$
L_D
=
-
\left(
\mathbb{E}_{x \sim p_{\text{data}}(x)} [\log D(x)]
+
\mathbb{E}_{z \sim p_z(z)} [\log(1 - D(G(z)))]
\right)
$$

---

## Generator Loss (Original)

$$
L_G
=
\mathbb{E}_{z \sim p_z(z)} [\log(1 - D(G(z)))]
$$

---

## Generator Loss (Non-Saturating Version)

$$
L_G
=
-
\mathbb{E}_{z \sim p_z(z)} [\log D(G(z))]
$$
