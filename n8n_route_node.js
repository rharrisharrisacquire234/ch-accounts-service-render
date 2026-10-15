// n8n Code node: "Route iXBRL or PDF"
// Place this straight after the document metadata HTTP Request.
// Use the is_ixbrl flag in an IF node to pick the branch.
//
// The zip check matters. Newer Companies House filings ship the tagged
// report inside a zip package rather than as a bare xhtml file, and the
// previous version of this node only looked for application/xhtml+xml, so
// every modern filing fell through to the expensive PDF branch.

const meta = $input.first().json;
const resources = meta.resources || {};
const types = Object.keys(resources);

const zipType = types.find(t => t.toLowerCase().includes('zip')) || null;
const hasXhtml = types.includes('application/xhtml+xml');
const hasIxbrl = Boolean(zipType) || hasXhtml;

return [{
  json: {
    is_ixbrl: hasIxbrl,
    document_url: (meta.links || {}).document || null,
    accept: zipType || (hasXhtml ? 'application/xhtml+xml' : 'application/pdf'),
    resources: types,
    pages: meta.pages || null,
    filename: meta.filename || null,
  }
}];
