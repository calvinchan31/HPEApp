# Contributing to HPE App

Thank you for your interest in contributing to HPE App! This document provides guidelines and instructions for contributing.

## Before You Start

Please review:
- [LICENSE](../LICENSE) — MIT License for app code
- [ACKNOWLEDGMENTS.md](../ACKNOWLEDGMENTS.md) — Third-party model and library licenses
- [PUBLISHING_CHECKLIST.md](../PUBLISHING_CHECKLIST.md) — Commercial use restrictions

## Development Setup

### Prerequisites
- macOS with Xcode 15+
- iPhone with A12 Bionic or newer (iOS 16+)
- XcodeGen: `brew install xcodegen`
- Python 3.8+ (for model export scripts)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/hpeapp.git
   cd hpeapp
   ```

2. **Generate Xcode project**
   ```bash
   xcodegen generate
   ```

3. **Open in Xcode**
   ```bash
   open HPEApp.xcodeproj
   ```

4. **Configure signing**
   - Select your physical iPhone as target (simulator doesn't support camera)
   - Go to Signing & Capabilities
   - Select your Team for code signing
   - Build & Run (⌘R)

##  Making Changes

### Code Style
- Use Swift 5.9+ syntax
- Follow Apple's Swift API Design Guidelines
- Use meaningful variable/function names
- Add comments for complex algorithms
- Include SPDX license headers on new files

### Commit Guidelines
- Write clear, descriptive commit messages
- Reference issues/PRs when applicable
- Keep commits focused on single features/fixes
- Format: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`

Example:
```bash
git commit -m "feat: add multi-person real-time tracking support"
git commit -m "fix: resolve memory leak in frame buffer"
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Test thoroughly on physical device
5. Commit with clear messages
6. Push to your fork
7. Open a Pull Request with:
   - Clear title describing the change
   - Description of what changed and why
   - Any related issues (closes #123)
   - Screenshots for UI changes

##  Testing

Before submitting a PR:
- Test on physical iPhone device
- Verify camera and motion sensors work
- Check FPS is reasonable (>4 FPS for live mode)
- Verify memory usage is stable
- Test with multiple lighting conditions

##  Reporting Issues

Use GitHub Issues with:
- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- Device model and iOS version
- Screenshots/videos if applicable

##  License Compliance

### For Code Contributions
- Your code will be licensed under MIT
- By contributing, you agree to this license

### For Model Modifications
- GVHMR, SMPL, and YOLOv8 have specific licenses
- Do not modify these without understanding the implications
- See [ACKNOWLEDGMENTS.md](../ACKNOWLEDGMENTS.md)

### Commercial Use Restrictions
- If you're commercializing HPE App:
  - **GVHMR**: Contact xwzhou@zju.edu.cn
  - **SMPL**: Contact smpl@tue.mpg.de
  - **YOLOv8**: Contact https://www.ultralytics.com/

##  Community Guidelines

- Be respectful and inclusive
- Provide constructive feedback
- Help other contributors
- Report security issues privately (don't open public issues)

##  Documentation

- User-facing docs: See [README.md](../README.md)
- Technical docs: See `docs/` folder
- Code comments: Use SwiftDoc format

##  Questions

- Check existing issues and discussions
- Review the technical documentation
- Open a discussion for architectural questions

##  Additional Resources

- [Apple Swift Style Guide](https://swift.org/documentation/articles/swift-api-design-guidelines)
- [Swift Concurrency](https://developer.apple.com/swift/concurrency/)
- [CoreML Documentation](https://developer.apple.com/documentation/coreml)
- Original [GVHMR Paper](https://github.com/zju3dv/GVHMR)

---

**Thank you for contributing to HPE App!** 
