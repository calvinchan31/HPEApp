import Foundation

struct SavedProcessedVideo: Identifiable, Codable {
    let id: String
    let createdAt: Date
    let title: String
    let sourceVideoName: String?
    let folderName: String
    let incamFileName: String?
    let globalFileName: String?
    let jsonFileName: String?
    let benchmarkFileName: String?
    let compareCompositeFileName: String?
    let sessionType: String?

    init(
        id: String = UUID().uuidString,
        createdAt: Date = Date(),
        title: String,
        sourceVideoName: String?,
        folderName: String,
        incamFileName: String?,
        globalFileName: String?,
        jsonFileName: String?,
        benchmarkFileName: String? = nil,
        compareCompositeFileName: String? = nil,
        sessionType: String? = nil
    ) {
        self.id = id
        self.createdAt = createdAt
        self.title = title
        self.sourceVideoName = sourceVideoName
        self.folderName = folderName
        self.incamFileName = incamFileName
        self.globalFileName = globalFileName
        self.jsonFileName = jsonFileName
        self.benchmarkFileName = benchmarkFileName
        self.compareCompositeFileName = compareCompositeFileName
        self.sessionType = sessionType
    }
}

final class SavedVideoLibrary {
    static let shared = SavedVideoLibrary()

    private init() {}

    private var rootURL: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docs.appendingPathComponent("GVHMRSavedVideos", isDirectory: true)
    }

    private var indexURL: URL {
        rootURL.appendingPathComponent("index.json")
    }

    func makeEntryDirectory() throws -> (folderName: String, folderURL: URL) {
        try ensureRootDirectory()
        let folderName = "save_\(Int(Date().timeIntervalSince1970))_\(UUID().uuidString.prefix(6))"
        let folderURL = rootURL.appendingPathComponent(folderName, isDirectory: true)
        try FileManager.default.createDirectory(at: folderURL, withIntermediateDirectories: true)
        return (folderName, folderURL)
    }

    func loadItems() -> [SavedProcessedVideo] {
        do {
            try ensureRootDirectory()
            guard FileManager.default.fileExists(atPath: indexURL.path) else { return [] }
            let data = try Data(contentsOf: indexURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            let decoded = try decoder.decode([SavedProcessedVideo].self, from: data)
            return decoded.sorted(by: { $0.createdAt > $1.createdAt })
        } catch {
            print("[SavedVideoLibrary] load failed: \(error)")
            return []
        }
    }

    func append(_ item: SavedProcessedVideo) {
        var items = loadItems()
        items.insert(item, at: 0)
        persist(items)
    }

    func delete(_ item: SavedProcessedVideo) {
        let folderURL = rootURL.appendingPathComponent(item.folderName, isDirectory: true)
        do {
            if FileManager.default.fileExists(atPath: folderURL.path) {
                try FileManager.default.removeItem(at: folderURL)
            }
        } catch {
            print("[SavedVideoLibrary] delete folder failed: \(error)")
        }

        var items = loadItems()
        items.removeAll(where: { $0.id == item.id })
        persist(items)
    }

    func resolvePreviewURL(for item: SavedProcessedVideo) -> URL? {
        let folderURL = entryDirectory(for: item)

        if let compare = item.compareCompositeFileName {
            let url = folderURL.appendingPathComponent(compare)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }

        if let incam = item.incamFileName {
            let url = folderURL.appendingPathComponent(incam)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        if let global = item.globalFileName {
            let url = folderURL.appendingPathComponent(global)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        return nil
    }

    func entryDirectory(for item: SavedProcessedVideo) -> URL {
        rootURL.appendingPathComponent(item.folderName, isDirectory: true)
    }

    func resolveIncamURL(for item: SavedProcessedVideo) -> URL? {
        guard let name = item.incamFileName else { return nil }
        let url = entryDirectory(for: item).appendingPathComponent(name)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    func resolveGlobalURL(for item: SavedProcessedVideo) -> URL? {
        guard let name = item.globalFileName else { return nil }
        let url = entryDirectory(for: item).appendingPathComponent(name)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    func resolveJSONURL(for item: SavedProcessedVideo) -> URL? {
        guard let name = item.jsonFileName else { return nil }
        let url = entryDirectory(for: item).appendingPathComponent(name)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    func resolveBenchmarkURL(for item: SavedProcessedVideo) -> URL? {
        guard let name = item.benchmarkFileName else { return nil }
        let url = entryDirectory(for: item).appendingPathComponent(name)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    func resolveCompareCompositeURL(for item: SavedProcessedVideo) -> URL? {
        guard let name = item.compareCompositeFileName else { return nil }
        let url = entryDirectory(for: item).appendingPathComponent(name)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    private func persist(_ items: [SavedProcessedVideo]) {
        do {
            try ensureRootDirectory()
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(items)
            try data.write(to: indexURL, options: .atomic)
        } catch {
            print("[SavedVideoLibrary] persist failed: \(error)")
        }
    }

    private func ensureRootDirectory() throws {
        try FileManager.default.createDirectory(at: rootURL, withIntermediateDirectories: true)
    }
}
