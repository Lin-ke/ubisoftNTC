Skip to main content

Shipping Neural Texture Compression in Assassin’s Creed Mirage
Article Poster
Assassin’s Creed Mirage
Shipping Neural Texture Compression in Assassin’s Creed Mirage
2026 February
Modern video games rely heavily on high-resolution physically based rendering (PBR) textures to deliver the visual quality and sharpness expected from AAA titles such as Assassin’s Creed. These high-resolution assets account for a large fraction of both disk and GPU memory usage (about 60% of disk size), and the growing demand for 4K textures is rapidly increasing overall game size. Widely used texture compression formats are no longer sufficient at this scale. Because they operate independently on each texture layer, they fail to exploit the correlations that exist across material channels. To address this limitation, we have developed a neural material texture compression technique that uses machine learning to exploit this cross-channel structure and reconstruct full PBR materials in real time, enabling higher effective compression (about 30% in Assassin’s Creed Mirage) without sacrificing visual quality.
Neural Texture Compression
image
Figure 1
The idea behind neural texture compression is simple: replace the conventional set of per-layer PBR textures with a latent representation shared across all material channels (fig 1), together with a small decoder network evaluated at sample time. The latent representation is stored as 2D textures, and the decoder reconstructs the material attributes (albedo, normal, roughness/metalness/AO, etc.) in the shader.
Neural Texture Encoding
A central design constraint is that the latent textures are exported as standard BC-compressed textures [1]. This preserves all the properties expected by the renderer, including random access, mipmapping, anisotropic filtering, and compatibility with existing streaming and residency systems. From the engine’s perspective, neural textures therefore behave like ordinary BC textures, with the only addition being a lightweight decode step in the shading stage, making integration straightforward. However, naively compressing the learned latent space is insufficient: BC compression artifacts would be amplified by the neural decoder at runtime.
Figure 2
To ensure that the learned representation faithfully matches runtime behavior, the training process explicitly models BC compression (fig. 2). BC decoding is simulated directly within the training graph. As a result, the network is optimized against the exact signal that will be sampled by the renderer, rather than against uncompressed neural features that would only be approximated by BC after training. Filtering is handled in the same way. Each training sample carries a continuous level-of-detail value, and the supervision signal is generated using the same trilinear or bicubic filtering behavior applied by the GPU. This causes the decoder to implicitly learn the filtering operation. At runtime, the renderer performs a single filtered fetch of the latent BC textures using the hardware sampler; the decoder then reconstructs the material channels directly from these filtered features. The output remains stable across mip transitions and under sub-pixel motion, without requiring any additional post-filtering passes.
Runtime reconstruction
Figure 3
At runtime, latent BC textures are sampled using the standard texture pipeline. The filtered latent features are passed through a small MLP with a single hidden layer, embedded in the pixel or G-buffer shader, to reconstruct the material attributes. Decoder parameters are stored in a small constant buffer or texture, and the reconstruction consists of a small number of matrix multiplications. Because the representation is BC-native, integration into the rendering pipeline is relatively direct.
Neural Texture Compression in AC Mirage
To ship in Assassin’s Creed Mirage, additional work was required to fully align Neural Texture Compression with the engine’s existing texture infrastructure. In particular, the technique was updated to obey the same streaming rules and mip-skip policies as traditional textures, ensuring correct behavior across various platforms. We also revised the storage format used for the latent representation. Instead of using the BC6 format as in our original paper [1], we stored the latent space using the BC1 format, as suggested by the paper Intel’s R&D team [2], which showed that BC1 is better suited for learned latent representations when compression efficiency is the primary constraint. With these adjustments, raw PBR texture sets could be exported directly from the engine and processed by the neural compression pipeline, without requiring changes to the material model, texture streaming system, or runtime sampling code.
Neural Texture Compression benefits from higher-resolution training targets. In a real-world production setting, however, high-quality reference textures are not always available. To handle these cases, we integrated an automatic material upscaling step [3] that synthesizes a higher-resolution training target from the available game-resolution textures whenever a true high-quality reference is missing. This upscaled signal is used as training supervision, allowing the network to learn a cleaner and more detailed reconstruction. The resulting latent representations are exported as BC-compressed textures using the same pipeline as standard assets. This avoids the need for hand-authored high-resolution replacements and ensures more consistent quality across materials with heterogeneous source data.
Figure 4
With the addition of automatic upscaling, the system forms a complete workflow capable of increasing the effective displayed resolution of material textures, even when higher-quality source data is unavailable, while remaining fully compatible with the engine’s existing texture and streaming infrastructure.
Below are screenshots comparing standard texture sets to neural textures. In these examples, neural textures not only preserve or improve visual quality, but also reduce memory usage by approximately 30%.
image
Figure 5
image
Figure 6
image
Figure 7
In the shipped game, usage was constrained by runtime performance. To keep costs within budget, Neural Texture Compression was applied selectively to a subset of assets, focusing on objects with high instance counts and significant memory pressure. This limited the total number of pixels reconstructed by the neural decoder, while still delivering a meaningful reduction in texture memory and preserving visual quality where it mattered most.
References
[1] Weinreich, Clément, Louis De Oliveira, Antoine Houdard, and Georges Nader. "Real‐Time Neural Materials using Block‐Compressed Features." In Computer Graphics Forum, vol. 43, no. 2.
[2] Laurent, Belcour, and Benyoub Anis. "Hardware Accelerated Neural Block Texture Compression with Cooperative Vectors." arXiv preprint arXiv:2506.06040 (2025).
[3] Du, Xin, Maoyuan Xu, and Zhi Ying. "MUJICA: Reforming SISR Models for PBR Material Super-Resolution via Cross-Map Attention." arXiv preprint arXiv:2508.09802 (2025).
Visit Other Ubisoft Channels
Download Ubisoft Connect

    Ubisoft Connect
    Support

    About us
    Investors
    Press

    Careers
    Locations
    More than games

    Privacy
    Terms of Use
    Notice at collection
    Refund policy
    Terms of Sale
    Withdrawal form
    Ubisoft+ Subscription Terms

    Interest-based Ads
    Do Not Sell / Share My Personal Information
    Limit Use / Disclosure of My Sensitive Personal Information

© 2026 Ubisoft Entertainment. All Rights Reserved. Ubisoft, Ubi.com and the Ubisoft logo are trademarks of Ubisoft Entertainment in the U.S. and/or other countries.
