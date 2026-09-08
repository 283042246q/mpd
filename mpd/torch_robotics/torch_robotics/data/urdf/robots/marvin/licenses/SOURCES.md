# Imported Marvin and Pika assets

The MPD repository's MIT license does not replace the upstream asset licenses.
This directory applies to both the original arm-only Marvin assets and the
additional dual-Pika assets. No copyright ownership is assigned by this import.

| Imported source | Attribution and declared license |
| --- | --- |
| https://github.com/Gabriel-Ning/marvin_description | Gabriel-Ning, as recorded in package.xml; package.xml declares Apache-2.0, but root LICENSE contains MIT-style permission text (unresolved discrepancy) |
| https://github.com/Gabriel-Ning/pika_gripper_description | Gabriel-Ning, as recorded in package.xml; ROS packaging Apache-2.0 |
| physical_ai_runtime/src/apps/marvin_mpd_bimanual_bringup | Gabriel-Ning, as recorded in package.xml; Apache-2.0 |
| https://github.com/agilexrobotics/pika_ros | Pika body geometry source identified by upstream docs/MESH_SOURCES.md; BSD declaration; upstream root LICENSE text retained |

`Apache-2.0.txt` contains the Apache 2.0 license text. The imported package.xml
files retain the original author, maintainer and license declarations under
`../sources/`. The original Pika `docs/MESH_SOURCES.md` is preserved there too:
the Pika body is frame-remapped AgileX geometry; the FinRay finger is package-local.
The upstream document does not separately identify copyright owners or licenses
for each package-local mesh; this import preserves that attribution boundary.

`pika_ros-BSD-3-Clause.txt` retains the upstream root LICENSE text (blank-line
trailing whitespace normalized), retrieved
2026-09-05 from https://raw.githubusercontent.com/agilexrobotics/pika_ros/master/LICENSE
(Git blob 5df67dcf95f92c0bf3ca46dda7aea51734818444). Its named copyright holder is
Tixiao Shan (2020); this is an upstream notice, not an inferred claim of ownership
over every Pika mesh. The Marvin root LICENSE is preserved byte-for-byte at
`../sources/marvin_description/LICENSE`: it contains MIT-style terms without
a named copyright line, inconsistent with its package.xml Apache declaration.
No standalone LICENSE/NOTICE was present in the local Pika description root.
These declarations do not establish a clear per-mesh redistribution grant:
seek upstream clarification for the Marvin discrepancy and package-local
FinRay/adaptor attribution before external redistribution. This migration
preserves available notices; it is not a licensing clearance.

Exact repository revisions, source/config hashes and each copied mesh hash are
recorded in `../pika_assets.lock.yaml`. Mesh bytes are copied unchanged. MPD
changes: ROS Xacro is expanded, mesh URIs are localized, the four Pika prismatic
joints are frozen with their origin translations adjusted, hardware tags are
omitted, and conservative collision envelopes are generated. Source Xacro and
configuration snapshots are retained unchanged for comparison.
