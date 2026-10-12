from pathlib import Path
import sys

p = Path('dynfed/selection.py')
if not p.exists():
    sys.exit('ERROR: Run from the DynFL repository root; dynfed/selection.py not found')
raw = p.read_bytes()
bom = raw.startswith(b'\xef\xbb\xbf')
text = raw.decode('utf-8-sig')
nl = '\r\n' if '\r\n' in text else '\n'
t = text.replace('\r\n','\n')
old1 = '''            for client_id in search_client_ids:
                current = evaluated.profile[client_id]
'''
new1 = '''            for client_id in search_client_ids:
                current = evaluated.profile[client_id]
                # In this base profile, the unchanged signature segments are
                # shared by every alternative candidate of this client.
                position = client_positions[client_id]
                signature_prefix = evaluated.profile_signature[:position]
                signature_suffix = evaluated.profile_signature[position + 1:]
'''
old2 = '''                    position = client_positions[client_id]
                    key = (
                        evaluated.profile_signature[:position]
                        + (token,)
                        + evaluated.profile_signature[position + 1:]
                    )
'''
new2 = '''                    key = signature_prefix + (token,) + signature_suffix
'''
if 'signature_prefix = evaluated.profile_signature[:position]' in t:
    sys.exit('ALREADY_PATCHED: no changes made')
if t.count(old1) != 1 or t.count(old2) != 1:
    sys.exit(f'ERROR: source pattern mismatch (loop={t.count(old1)}, key={t.count(old2)}); no changes made')
import datetime
backup = p.with_name(p.name + '.bak_signature_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
backup.write_bytes(raw)
updated = t.replace(old1,new1,1).replace(old2,new2,1).replace('\n',nl).encode('utf-8')
if bom: updated = b'\xef\xbb\xbf' + updated
try:
    compile(updated.decode('utf-8-sig'),str(p),'exec')
except Exception:
    backup.unlink(missing_ok=True)
    raise
p.write_bytes(updated)
print('PATCH_OK',p)
print('BACKUP',backup)
