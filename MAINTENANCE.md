# Maintenance Guide

This document outlines the maintenance procedures for the Awesome Mistral repository.

## Monthly Maintenance Checklist

Perform these tasks on the first week of each month:

### Link Verification
- [ ] Review automated link checker results from GitHub Actions
- [ ] Fix or remove any dead links
- [ ] Update redirected links to their final destinations

### Content Updates
- [ ] Check [Mistral AI Blog](https://mistral.ai/news/) for new announcements
- [ ] Review [Mistral AI GitHub](https://github.com/mistralai) for new repositories
- [ ] Check [Hugging Face trending models](https://huggingface.co/models?sort=trending) for popular Mistral fine-tunes
- [ ] Review open issues and PRs

### Quality Control
- [ ] Verify star counts on major projects are still accurate (±20%)
- [ ] Archive or remove deprecated/unmaintained projects
- [ ] Ensure all descriptions are accurate and up-to-date

## Adding New Resources

When adding new resources, follow these guidelines:

### Format
```markdown
- 🌍 [Project Name](url) ⭐ 10k+ – Brief, technical description.
```

### Quality Criteria
1. **Active maintenance**: Last commit within 6 months (preferably 3)
2. **Documentation**: Clear README with usage instructions
3. **Adoption**: Minimum 100 stars for tools, 500 for major projects
4. **Relevance**: Must specifically support or integrate with Mistral models

### Star Count Guidelines
Only include star counts for projects with 5k+ stars:
- ⭐ 5k+
- ⭐ 10k+
- ⭐ 20k+
- ⭐ 50k+
- ⭐ 100k+

## Commit Message Convention

Use these prefixes for commits:

```
add: [Resource Name] to [Section]
remove: [Resource Name] (reason)
update: [Resource Name] description/link
fix: broken link for [Resource Name]
docs: update maintenance guide
chore: update star counts
```

## Release Notes

When making significant updates, create a release with:
- Summary of additions
- Summary of removals
- Any structural changes

## Automation

### GitHub Actions
- **Link Checker**: Runs weekly on Sundays
- Creates issues automatically for dead links

### Manual Triggers
To run the link checker manually:
1. Go to Actions tab
2. Select "Check Links" workflow
3. Click "Run workflow"

## Contact

For questions about maintenance procedures, open a discussion in the repository.
