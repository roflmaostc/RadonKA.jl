# Mathematial background 
## Introduction
The Radon transform [radon1917uber](@cite) is an integral transform which is the foundation
behind computed tomography (CT).
It projects the values of a function along straight lines onto a detector.
These projections are calculated for a set of different observation angles around the function.
Such a dataset allows to reconstruct the original function distribution.



## Definition
The general definition of the parallel attenuated Radon transform (ART) $\mathcal{R}_{\mu}$ is

$$g(s, \theta) = (\mathcal{R}_{\mu} f)(s, \theta) = \int_{-\infty}^{\infty} f(s\,\pmb{\omega} + t\,\pmb{\omega}^{\perp}) \exp\left(- \int_t^{\infty} \mu\left(s\,\pmb{\omega} + \tau\,\pmb{\omega}^{\perp}\right)\,\,\mathrm{d}\tau\right)\,\, \mathrm{d}t$$

where $\mu$ is the position dependent absorption coefficient. The vectors are defined as $\omega=(\cos \theta, \sin \theta)$ and
$\omega^{\perp}=(-\sin \theta, \cos \theta)$.

In case of zero absorption ($\mu=0$) the ART reduces to the ordinary Radon transform.
## Adjoint
If $\mu$ is known, the adjoint of the ART with respect to $f$ is given by

$$(\mathcal{R}_{\mu}^* g)(x, y) = \int_{0}^{2\pi} g( \pmb{\omega} \cdot [x,y]^{\intercal}, \theta) \exp\left(- \int_{[x,y]^{\intercal} \cdot \pmb{\omega}^{\perp}}^{\infty} \mu\left([x,y]^{\intercal} \cdot \pmb{\omega} + \tau\,\pmb{\omega}^{\perp}\right)\,\,\mathrm{d}\tau\right)\,\, \mathrm{d}\theta$$

where $\mu$ is the position dependent absorption coefficient.
Qualitatively, the Radon transform calculates the _shadow_ of a sample if illuminated with plane waves.


## Discretization
On the other hand, the adjoint Radon transform _smears_ the _shadows_ (projection patterns) back into space and deposits values at each point.
The pair of $\mathcal{R}_{\mu}$ and $\mathcal{R}_{\mu}^*$ can be used in gradient descent based optimizers.

In the figure below we show the geometry to the presented equations. Further, we show how the grid is discretized.
`RadonKA.jl` traces a ray through the pixel grid.
The function value it accumulates (Radon transform) or deposits (adjoint Radon transform) depends on the intersection length with each pixel.

```@raw html
<img src="../assets/geometry.png" alt="The geometry of the Radon transform." width="500"/>
```


## Applications
The ordinary Radon transform is the mathematical concept behind CT [Buzug2011](@cite). CT allows to recover a 3D absorption map of objects such as human bodies.
One application of the ART is single photon emission computerized tomography (SPECT) [spect](@cite).
In SPECT a tracer is injected in the specimen (e.g. blood of human bodies). The trace is radioactive and emits gamma rays which are detected.
Since the gamma rays are created inside the body, its intensity is weakened from the starting point of the emission until hitting the detector.
A more recent usage of the ART is within the context of Tomographic Volumetric Additive Manufacturing (TVAM) [Kelly_Bhattacharya_Heidari_Shusteff_Spadaccini_Taylor_2019](@cite), [Loterie_Delrot_Moser_2020](@cite).

In TVAM a photosensitive resin is illuminated with projection patterns from different angular positions.
The light is absorbed while propagating through that medium and reacting with the photo-initiator. Once a certain voxel in space received a
sufficient energy dose, it is going to polymerize. The mathematical process behind this printer is given by the adjoint of the ART.

## References

```@bibliography
```
