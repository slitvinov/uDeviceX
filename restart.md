#  file format

Restart stores simulation variables of solvent, wall, rbcs and rigid bodies under the following file naming:

```
strt/[code]/[XXX].[YYY].[ZZZ]/[ttt].[ext]
```
where:
* `[code]` is `flu` (solvent), `wall`, `rbc` or `rig` (rigid bodies)
* `[XXX].[YYY].[ZZZ]` are the coordinates of the processor (no dir if single processor)
* `[ttt]` is the id of the restart; it has a magic value `final` for the last step of simulation and is used to start from there. 
* `[ext]` is the extension of the file: `bop` for particles, `ss` for solids, `id.bop` for particle ids

Special case:
```
strt/[code]/[magic name].[ext]
```
example: template frozen particles from rigid bodies: `[magic name]` = frozen
