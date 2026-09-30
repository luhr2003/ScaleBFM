"""Build a variant of the ScaleTrack G1 asset whose feet use MagicSim's collision model instead of the seven thin capsules.

MagicSim's `g1_new.usd` models every foot with two flat cylinders ("discs") in the ankle_roll_link frame (front: r=0.033 m at x=+0.1089,
rear: r=0.030 m at x=-0.0355, height 0.015 m, axis z, z=-0.0275). The variant references the original asset unchanged (articulation, joints,
masses, inertias are identical, the link masses and centres of mass of the two assets agree), switches the capsules off and adds the two
discs, so the policy interface is the same. Each foot keeps seven collision shapes (two capsule prims are removed, five stay but no longer
collide) because Isaac Lab's physics-material randomization assumes the same shape count in every environment when assets are mixed.

Use it for training / evaluation with `SCALETRACK_ROBOT_USD=<path of the generated .usda>`, or mix it with the original asset per environment
with `SCALETRACK_ROBOT_USD_MIX=<original.usda>,<variant.usda>`.

usage: python scripts/assets/make_discfeet_usd.py --headless     (needs an Isaac Lab python: pxr is only available inside the Kit app)
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

ASSET_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "source", "scaletrack", "scaletrack", "assets", "robots")
)
SRC = os.path.join(ASSET_ROOT, "g1_29dof", "g1_29dof.usda")
OUT = os.path.join(ASSET_ROOT, "g1_29dof_discfeet", "g1_29dof_discfeet.usda")
ROOT = "/g1_sphere_capsule_complex"
# MagicSim g1_new.usd: (name, x, y, z, radius); height 0.015 m, axis Z, in the ankle_roll_link frame
DISCS = {
    "left": [("front", 0.1089, 0.0001, -0.0275, 0.033), ("rear", -0.0355, 0.0002, -0.0275, 0.030)],
    "right": [("front", 0.1089, -0.0001, -0.0275, 0.033), ("rear", -0.0355, -0.0002, -0.0275, 0.030)],
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
if os.path.exists(OUT):
    os.remove(OUT)
stage = Usd.Stage.CreateNew(OUT)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
root = stage.DefinePrim(ROOT, "Xform")
root.GetReferences().AddReference(os.path.relpath(SRC, os.path.dirname(OUT)))  # relative: the repository stays relocatable
stage.SetDefaultPrim(root)


def uninstance(path):
    """Prims of an instanceable asset are instance proxies (read-only): switch instancing off along the path, top down."""
    parts = path.strip("/").split("/")
    for k in range(1, len(parts) + 1):
        prim = stage.OverridePrim("/" + "/".join(parts[:k]))
        if prim.IsInstanceable():
            prim.SetInstanceable(False)


for side, discs in DISCS.items():
    collisions = f"{ROOT}/pelvis/{side}_ankle_roll_link/collisions"
    uninstance(collisions)
    for i in (1, 7):
        stage.OverridePrim(f"{collisions}/{side}_foot{i}_collision").SetActive(False)
    for i in range(2, 7):
        capsule = stage.OverridePrim(f"{collisions}/{side}_foot{i}_collision/{side}_foot{i}_collision")
        UsdPhysics.CollisionAPI.Apply(capsule).CreateCollisionEnabledAttr(False)
    for name, x, y, z, radius in discs:
        xform = UsdGeom.Xform.Define(stage, f"{collisions}/{side}_foot_disc_{name}")
        UsdGeom.XformCommonAPI(xform).SetTranslate(Gf.Vec3d(x, y, z))
        cylinder = UsdGeom.Cylinder.Define(stage, f"{collisions}/{side}_foot_disc_{name}/{side}_foot_disc_{name}")
        cylinder.CreateRadiusAttr(radius)
        cylinder.CreateHeightAttr(0.015)
        cylinder.CreateAxisAttr("Z")
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
stage.GetRootLayer().Save()

check = Usd.Stage.Open(OUT)
for prim in Usd.PrimRange(check.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
    if "left_ankle_roll_link/collisions" in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.CollisionAPI):
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        print("collider:", str(prim.GetPath()).split("pelvis/")[-1], prim.GetTypeName(), "enabled" if enabled is not False else "DISABLED", flush=True)
print("wrote", OUT, flush=True)
os._exit(0)
