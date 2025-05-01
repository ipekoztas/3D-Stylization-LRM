# 3D Stylization via Large Reconstruction Model

<div class="is-size-5 publication-authors">
    <span class="author-block">
    <a href="https://ipekoztas.github.io/">İpek Öztaş</a><sup>1</sup>,</span>
    <span class="author-block">
    <a href="https://www.duygu-ceylan.com/">Duygu Ceylan</a><sup>2</sup>,
    </span>
    <span class="author-block">
    <a href="http://www.cs.bilkent.edu.tr/~adundar/">Aysegul Dundar</a><sup>1</sup>
    </span>
</div>

<div class="is-size-5 publication-authors">
  <span class="author-block"><sup>1</sup>Bilkent University,&nbsp;&nbsp;</span>
  <span class="author-block"><sup>2</sup>Adobe Research&nbsp;&nbsp;</span>
</div>

<a href="https://arxiv.org/abs/2504.21836"><img src="https://img.shields.io/badge/arXiv-2311.03335-b31b1b.svg" height=22.5></a>
<a href="https://ipekoztas.github.io/3DStylizationLRM/"><img src="https://img.shields.io/static/v1?label=Project&message=Page&color=red" height=20.5></a>

## Abstract

With the growing success of text or image guided 3D generators, users demand more control over the generation process, appearance stylization being one of them. Given a reference image, this requires adapting the appearance of a generated 3D asset to reflect the visual style of the reference while maintaining visual consistency from multiple viewpoints. To tackle this problem, we draw inspiration from the success of 2D stylization methods that leverage the attention mechanisms in large image generation models to capture and transfer visual style. In particular, we probe if large reconstruction models, commonly used in the context of 3D generation, has a similar capability. We discover that the certain attention blocks in these models capture the appearance specific features. By injecting features from a visual style image to such blocks, we develop a simple yet effective 3D appearance stylization method. 
Our method does not require training or test time optimization. Through both quantitative and qualitative evaluations, we demonstrate that our approach achieves superior results in terms of 3D appearance stylization, significantly improving efficiency while maintaining high-quality visual outcomes.


## Results
<table>
    <tr>
        <td></td>
        <td><img src='teaser/original/fish.png' alt='fish.png'></td>
        <td><img src='teaser/original/fox.png' alt='fox.png'></td>
        <td><img src='teaser/original/genshin_building.png' alt='genshin_building.png'></td>
        <td><img src='teaser/original/train.png' alt='train.png'></td>
    </tr>
    <tr>
        <td><img src='teaser/styles/leo.jpg'></td>
        <td><img src='teaser/ours/fish/leo/fish.gif' alt='fish.gif'></td>
        <td><img src='teaser/ours/fox/leo/fox.gif' alt='fox.gif'></td>
        <td><img src='teaser/ours/genshin_building/leo/genshin_building.gif' alt='genshin_building.gif'></td>
        <td><img src='teaser/ours/train/leo/train.gif' alt='train.gif'></td>
    </tr>
    <tr>
        <td><img src='teaser/styles/snake.jpg'></td>
        <td><img src='teaser/ours/fish/snake/fish.gif' alt='fish.gif'></td>
        <td><img src='teaser/ours/fox/snake/fox.gif' alt='fox.gif'></td>
        <td><img src='teaser/ours/genshin_building/snake/genshin_building.gif' alt='genshin_building.gif'></td>
        <td><img src='teaser/ours/train/snake/train.gif' alt='train.gif'></td>
    </tr>
    <tr>
        <td><img src='teaser/styles/starry_night.jpg'></td>
        <td><img src='teaser/ours/fish/starry_night/fish.gif' alt='fish.gif'></td>
        <td><img src='teaser/ours/fox/starry_night/fox.gif' alt='fox.gif'></td>
        <td><img src='teaser/ours/genshin_building/starry_night/genshin_building.gif' alt='genshin_building.gif'></td>
        <td><img src='teaser/ours/train/starry_night/train.gif' alt='train.gif'></td>
    </tr>
</table>
We introduce the use of large pre-trained reconstruction
models for 3D stylization, providing a novel approach to
transferring artistic styles to 3D objects.
Our framework achieves 3D stylization without the need for
training or test-time optimization. This makes our approach
practical for interactive applications.
