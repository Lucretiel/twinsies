# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 1.2.0

### Added

- Add a feature flag, `safest-memory-ordering`, which upgrades a few memory orderings to pass miri. It's a point of ongoing investigation to determine if those orderings are necessary, or if this is a false positive from miri. This feature is enabled by default.

## 1.1.0

### Changed

- Remove `T: Debug` requirement from `impl<T> Debug for Joint<T>`. The actual output is unchanged.

## 1.0.0

Initial version!
