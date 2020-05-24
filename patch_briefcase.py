from briefcase.platforms.macOS import dmg

with open(dmg.__file__, 'r') as f:
    source = f.readlines()

lineno = source.index('        self.dmgbuild.build_dmg(\n')
source.insert(
    lineno, '        dmg_settings["size"] = os.environ.get("DMGSIZE", None)\n'
)
source.insert(lineno, '        import os\n')
with open(dmg.__file__, 'w') as f:
    f.write("".join(source))
